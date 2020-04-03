// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;

namespace Microsoft.ML.Transforms
{

    internal sealed class ForecastingPivotFeaturizerDataView : IDataTransform
    {
        #region Typed Columns
        private ForecastingPivotTransformer _parent;

        #endregion

        private readonly IHostEnvironment _host;
        private readonly IDataView _source;
        private readonly string[] _columnsToPivot;
        private readonly DataViewSchema _schema;

        internal ForecastingPivotFeaturizerDataView(IHostEnvironment env, IDataView input, string[] columnsToPivot, ForecastingPivotTransformer parent)
        {
            _host = env;
            _source = input;

            _columnsToPivot = columnsToPivot;
            _parent = parent;

            // One new column is added per ColumnsToPivot passed in.
            // Rows are also added
            //TODO: Add correct mapping logic here.
            _schema = _source.Schema;
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema => _schema;

        public IDataView Source => _source;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValueOrNull(rand);

            var input = _source.GetRowCursorForAllColumns();
            return new Cursor(_host, input, _columnsToPivot, _schema);
        }

        // Can't use parallel cursors so this defaults to calling non-parallel version
        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) =>
             new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };

        // Since we may delete rows we don't know the row count
        public long? GetRowCount() => null;

        public void Save(ModelSaveContext ctx)
        {
            _parent.Save(ctx);
        }

        private sealed class Cursor : DataViewRowCursor
        {
            private class PivotColumn
            {
                private ValueGetter<VBuffer<double>> _getter;
                private VBuffer<double> _curValues;
                private ImmutableArray<int> _dimensions;

                private int _curRow;
                private int _curCol;

                public PivotColumn(DataViewRowCursor input, string columnName)
                {
                    _getter = input.GetGetter<VBuffer<double>>(input.Schema[columnName]);
                    _curValues = default;
                    _dimensions = ((VectorDataViewType)(input.Schema[columnName].Type)).Dimensions;
                    ColumnCount = _dimensions[1];

                    _curRow = 0;
                    _curCol = 0;
                }

                public int RowCount => _dimensions[0];

                public int ColumnCount { get; private set; }

                public double GetValueAndSetRowColumn(int row, int col)
                {
                    _curRow = row;
                    _curCol = col;
                    return GetValue(_curRow, _curCol);
                }

                public void MoveNext()
                {
                    _getter(ref _curValues);
                }

                private double GetValue(int row, int col)
                {
                    return _curValues.GetItemOrDefault((row * ColumnCount) + col);
                }

                public double GetStoredValue()
                {
                    return GetValue(_curRow, _curCol);
                }
            }

            private readonly IChannelProvider _ch;
            private DataViewRowCursor _input;
            private long _position;
            private bool _isGood;
            private readonly string[] _columnsToPivot;
            private readonly DataViewSchema _schema;

            private Dictionary<string, PivotColumn> _pivotColumns;
            private readonly int _maxCols;
            private int _currentCol;

            public Cursor(IChannelProvider provider, DataViewRowCursor input, string[] columnsToPivot, DataViewSchema schema)
            {
                _ch = provider;
                _ch.CheckValue(input, nameof(input));

                // Start is good at true. This is not exposed outside of the class.
                _isGood = true;
                _input = input;
                _position = -1;
                _schema = schema;

                _pivotColumns = new Dictionary<string, PivotColumn>();
                _currentCol = -1;

                _columnsToPivot = columnsToPivot;

                foreach (var col in columnsToPivot)
                {
                    _pivotColumns[col] = new PivotColumn(input, col);
                }

                // All columns should have the same amount of slots in the vector
                _maxCols = _pivotColumns.First().Value.ColumnCount;
            }

            public sealed override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                       (ref DataViewRowId val) =>
                       {
                           _ch.Check(_isGood, RowCursorUtils.FetchValueStateError);
                           val = new DataViewRowId((ulong)Position, 0);
                       };
            }

            public sealed override DataViewSchema Schema => _schema;

            /// <summary>
            /// Since rows will be dropped
            /// </summary>
            public override bool IsColumnActive(DataViewSchema.Column column) => true;

            /// <summary>
            /// Returns a value getter delegate to fetch the value of column with the given columnIndex, from the row.
            /// This throws if the column is not active in this row, or if the type
            /// <typeparamref name="TValue"/> differs from this column's type.
            /// </summary>
            /// <typeparam name="TValue"> is the column's content type.</typeparam>
            /// <param name="column"> is the output column whose getter should be returned.</param>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                _ch.Check(IsColumnActive(column));

                if (_columnsToPivot.Contains(column.Name))
                {
                    return MakeGetter(_input, column) as ValueGetter<TValue>;
                }
                else
                {
                    return _input.GetGetter<TValue>(column);
                }
            }

            public override bool MoveNext()
            {
                bool exitLoop = false;
                while (_isGood && !exitLoop)
                {
                    // Make sure that we advance our source pointer if necesary.
                    if (_currentCol == _maxCols || _currentCol == -1)
                        _isGood = _input.MoveNext();

                    if (!_isGood)
                        break;

                    for (int col = _currentCol; col < _maxCols; col++)
                    {
                        var nanFound = false;

                        foreach (var column in _pivotColumns)
                        {
                            for (int row = 0; row < column.Value.RowCount; row++)
                            {
                                if (double.IsNaN(column.Value.GetValueAndSetRowColumn(row, col)))
                                {
                                    nanFound = true;
                                    break;
                                }
                            }

                            if (nanFound)
                                break;
                        }

                        // Break from loop because we now have valid values
                        // Update the _currentCol so we start from the correct place next time.
                        // Update our current position.
                        if (!nanFound)
                        {
                            exitLoop = true;
                            _currentCol = col + 1;
                            _position++;
                            break;
                        }
                    }
                }

                return _isGood;
            }

            public sealed override long Position => _position;

            public sealed override long Batch => _input.Batch;

            private Delegate MakeGetter(DataViewRow input, DataViewSchema.Column column)
            {
                // TODO: wrapper

                ValueGetter<double> result = (ref double dst) => {
                    dst = _pivotColumns[column.Name].GetStoredValue();
                };

                return result;
            }

        }
    }
}
