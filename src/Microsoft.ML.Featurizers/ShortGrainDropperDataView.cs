// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Runtime;
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;
using static Microsoft.ML.Featurizers.ShortGrainDropperEstimator;

namespace Microsoft.ML.Transforms
{

    internal sealed class ShortGrainDropperDataView : IDataTransform
    {
        #region Typed Columns
        private ShortDropTransformer _parent;
        public class SharedColumnState
        {
            public SharedColumnState()
            {
                SourceCanMoveNext = true;
                MemStream = new MemoryStream(4096);
                BinWriter = new BinaryWriter(MemStream, Encoding.UTF8);
            }

            public bool SourceCanMoveNext { get; set; }
            public int TransformedDataPosition { get; set; }

            // Hold the serialized data that we are going to send to the native code for processing.
            public MemoryStream MemStream { get; set; }
            public BinaryWriter BinWriter { get; set; }
        }

        private abstract class TypedColumn
        {
            private protected SharedColumnState SharedState;

            internal readonly DataViewSchema.Column Column;
            internal TypedColumn(DataViewSchema.Column column, SharedColumnState state)
            {
                Column = column;
                SharedState = state;
            }

            internal abstract void InitializeGetter(DataViewRowCursor cursor, TransformerEstimatorSafeHandle transformerParent,
                string[] grainColumns, Dictionary<string, TypedColumn> allColumns);

            internal abstract TypeId GetTypeId();
            internal abstract void SerializeValue(BinaryWriter binaryWriter);

            public bool MoveNext(DataViewRowCursor cursor)
            {
                SharedState.TransformedDataPosition++;

                // TODO: Here is where we will call the transformer and move until we get true back.

                //if (SharedState.TransformedData == null || SharedState.TransformedDataPosition >= SharedState.TransformedData.Length)
                //    SharedState.SourceCanMoveNext = cursor.MoveNext();

                //if (!SharedState.SourceCanMoveNext)
                //    if (SharedState.TransformedDataPosition >= SharedState.TransformedData.Length)
                //    {
                //        if (!SharedState.TransformedDataHandler.IsClosed)
                //            SharedState.TransformedDataHandler.Dispose();
                //        return false;
                //    }

                return true;
            }

            internal static TypedColumn CreateTypedColumn(DataViewSchema.Column column, string[] columns, SharedColumnState state)
            {
                var type = column.Type.RawType.ToString();
                if (type == typeof(ReadOnlyMemory<char>).ToString())
                    return new StringTypedColumn(column, state);

                throw new InvalidOperationException($"Unsupported type {type}. Grains can only be of type string");
            }
        }

        private abstract class TypedColumn<T> : TypedColumn
        {
            private ValueGetter<T> _getter;
            private ValueGetter<T> _sourceGetter;
            private long _position;

            internal TypedColumn(DataViewSchema.Column column, SharedColumnState state) :
                base(column, state)
            {
                _position = -1;
            }

            internal override unsafe void InitializeGetter(DataViewRowCursor cursor, TransformerEstimatorSafeHandle transformerParent,
                string[] grainColumns, Dictionary<string, TypedColumn> allColumns)
            {
                _sourceGetter = cursor.GetGetter<T>(Column);

                _getter = (ref T dst) =>
                {
                    dst = default;

                    // TODO: Wrapper
                    // this is so it will build while we are just doing the wrapper.
                    if(_position == -1)
                    {
                        // dummy check so _position is used.
                    }

                    //IntPtr errorHandle = IntPtr.Zero;
                    //bool success = false;
                    //if (SharedState.TransformedData == null || SharedState.TransformedDataPosition >= SharedState.TransformedData.Length)
                    //{
                    //    // Free native memory if we are about to get more
                    //    if (SharedState.TransformedData != null && SharedState.TransformedDataPosition >= SharedState.TransformedData.Length)
                    //        SharedState.TransformedDataHandler.Dispose();

                    //    var outputDataSize = IntPtr.Zero;
                    //    NativeBinaryArchiveData* outputData = default;
                    //    while (outputDataSize == IntPtr.Zero && SharedState.SourceCanMoveNext)
                    //    {
                    //        BuildColumnByteArray(allColumns, allImputedColumnNames);
                    //        QueueDataForNonImputedColumns(allColumns, allImputedColumnNames);
                    //        fixed (byte* bufferPointer = SharedState.MemStream.GetBuffer())
                    //        {
                    //            var binaryArchiveData = new NativeBinaryArchiveData() { Data = bufferPointer, DataSize = new IntPtr(SharedState.MemStream.Position) };
                    //            success = TransformDataNative(transformer, binaryArchiveData, out outputData, out outputDataSize, out errorHandle);
                    //            if (!success)
                    //                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));
                    //        }

                    //        if (outputDataSize == IntPtr.Zero)
                    //            SharedState.SourceCanMoveNext = cursor.MoveNext();

                    //        SharedState.MemStream.Position = 0;
                    //    }

                    //    if (!SharedState.SourceCanMoveNext)
                    //        success = FlushDataNative(transformer, out outputData, out outputDataSize, out errorHandle);

                    //    if (!success)
                    //        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    //    if (outputDataSize.ToInt32() > 0)
                    //    {
                    //        SharedState.TransformedDataHandler = new TransformedDataSafeHandle((IntPtr)outputData, outputDataSize);
                    //        SharedState.TransformedData = new NativeBinaryArchiveData[outputDataSize.ToInt32()];
                    //        for (int i = 0; i < outputDataSize.ToInt32(); i++)
                    //        {
                    //            SharedState.TransformedData[i] = *(outputData + i);
                    //        }
                    //        SharedState.TransformedDataPosition = 0;
                    //    }
                    //}

                    //// Base case where we didn't impute the column
                    //if (!allImputedColumnNames.Contains(Column.Name))
                    //{
                    //    var imputedData = SharedState.TransformedData[SharedState.TransformedDataPosition];
                    //    // If the row was imputed we want to just return the default value for the type.
                    //    if (BoolTypedColumn.GetBoolFromNativeBinaryArchiveData(imputedData.Data, 0))
                    //    {
                    //        dst = default;
                    //    }
                    //    else
                    //    {
                    //        // If the row wasn't imputed, get the original value for that row we stored in the queue and return that.
                    //        if (_position != cursor.Position)
                    //        {
                    //            _position = cursor.Position;
                    //            _cache = SourceQueue.Dequeue();
                    //        }
                    //        dst = _cache;
                    //    }
                    //}
                    //// If we did impute the column then parse the data from the returned byte array.
                    //else
                    //{
                    //    var imputedData = SharedState.TransformedData[SharedState.TransformedDataPosition];
                    //    int offset = 0;
                    //    foreach (var columnName in allImputedColumnNames)
                    //    {
                    //        var col = allColumns[columnName];
                    //        if (col.Column.Name == Column.Name)
                    //        {
                    //            dst = GetDataFromNativeBinaryArchiveData(imputedData.Data, offset);
                    //            return;
                    //        }

                    //        offset += col.GetDataSizeInBytes(imputedData.Data, offset);
                    //    }

                    //    // This should never be hit.
                    //    dst = default;
                    //}
                };
            }

            private void BuildColumnByteArray(Dictionary<string, TypedColumn> allColumns, string[] columns)
            {
                foreach (var column in columns)
                {
                    allColumns[column].SerializeValue(SharedState.BinWriter);
                }
            }

            private protected T GetSourceValue()
            {
                T value = default;
                _sourceGetter(ref value);
                return value;
            }

            internal override TypeId GetTypeId()
            {
                return typeof(T).GetNativeTypeIdFromType();
            }

            internal unsafe abstract T GetDataFromNativeBinaryArchiveData(byte* data, int offset);
        }

        private class StringTypedColumn : TypedColumn<ReadOnlyMemory<char>>
        {
            internal StringTypedColumn(DataViewSchema.Column column, SharedColumnState state) :
                base(column, state)
            {
            }

            internal override void SerializeValue(BinaryWriter binaryWriter)
            {
                var value = GetSourceValue().ToString();
                var stringBytes = Encoding.UTF8.GetBytes(value);

                binaryWriter.Write(stringBytes.Length);

                binaryWriter.Write(stringBytes);
            }

            internal unsafe override ReadOnlyMemory<char> GetDataFromNativeBinaryArchiveData(byte* data, int offset)
            {
                var size = *(uint*)(data + offset);

                var bytes = new byte[size];
                Marshal.Copy((IntPtr)(data + offset + sizeof(uint)), bytes, 0, (int)size);
                return Encoding.UTF8.GetString(bytes).AsMemory();
            }
        }

        #endregion

        #region Native Exports

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_BinaryArchive_Transform"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool TransformDataNative(TransformerEstimatorSafeHandle transformer, /*in*/ NativeBinaryArchiveData data, out NativeBinaryArchiveData* outputData, out IntPtr outputDataSize, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_BinaryArchive_Transform"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool FlushDataNative(TransformerEstimatorSafeHandle transformer, out NativeBinaryArchiveData* outputData, out IntPtr outputDataSize, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperFeaturizer_BinaryArchive_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
        private static extern unsafe bool DestroyTransformedDataNative(IntPtr data, IntPtr dataSize, out IntPtr errorHandle);

        #endregion

        #region Native SafeHandles

        internal class TransformedDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
        {
            private IntPtr _size;
            public TransformedDataSafeHandle(IntPtr handle, IntPtr size) : base(true)
            {
                SetHandle(handle);
                _size = size;
            }

            protected override bool ReleaseHandle()
            {
                // Not sure what to do with error stuff here.  There shouldn't ever be one though.
                return DestroyTransformedDataNative(handle, _size, out IntPtr errorHandle);
            }
        }

        #endregion

        private readonly IHostEnvironment _host;
        private readonly IDataView _source;
        private readonly string[] _grainColumns;
        private readonly DataViewSchema _schema;

        internal ShortGrainDropperDataView(IHostEnvironment env, IDataView input, string[] grainColumns, ShortDropTransformer parent)
        {
            _host = env;
            _source = input;

            _grainColumns = grainColumns;
            _parent = parent;

            // Use existing schema since it doesn't change.
            _schema = _source.Schema;
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema => _schema;

        public IDataView Source => _source;

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.AssertValueOrNull(rand);

            var input = _source.GetRowCursorForAllColumns();
            return new Cursor(_host, input, _parent.CloneTransformer(), _grainColumns, _schema);
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
            private readonly IChannelProvider _ch;
            private DataViewRowCursor _input;
            private long _position;
            private bool _isGood;
            private readonly Dictionary<string, TypedColumn> _grainColumns;
            private readonly DataViewSchema _schema;
            private readonly TransformerEstimatorSafeHandle _transformer;

            public Cursor(IChannelProvider provider, DataViewRowCursor input, TransformerEstimatorSafeHandle transformer, string[] grainColumns, DataViewSchema schema)
            {
                _ch = provider;
                _ch.CheckValue(input, nameof(input));

                _input = input;
                var length = input.Schema.Count;
                _position = -1;
                _schema = schema;
                _transformer = transformer;

                var sharedState = new SharedColumnState();

                _grainColumns = _schema.Select(x => TypedColumn.CreateTypedColumn(x, grainColumns, sharedState)).ToDictionary(x => x.Column.Name); ;

                foreach (var column in _grainColumns.Values)
                {
                    column.InitializeGetter(_input, transformer, grainColumns, _grainColumns);
                }
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

            protected override void Dispose(bool disposing)
            {
                if (!_transformer.IsClosed)
                    _transformer.Close();
            }

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

                return _input.GetGetter<TValue>(column);
            }

            public override bool MoveNext()
            {
                _position++;
                _isGood = _input.MoveNext();//_grainColumns[IsRowImputedColumnName].MoveNext(_input);
                return _isGood;
            }

            public sealed override long Position => _position;

            public sealed override long Batch => _input.Batch;
        }
    }
}
