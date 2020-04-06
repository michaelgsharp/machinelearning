using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Featurizers.CommonExtensions;
using static Microsoft.ML.SchemaShape.Column;

[assembly: LoadableClass(typeof(ForecastingPivotTransformer), null, typeof(SignatureLoadModel),
    ForecastingPivotTransformer.UserName, ForecastingPivotTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(ForecastingPivotTransformer), null, typeof(SignatureLoadDataTransform),
   ForecastingPivotTransformer.UserName, ForecastingPivotTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(ForecastingPivotTransformerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class ForecastingPivotFeaturizerExtensionClass
    {

        /// <summary>
        /// Drops rows when there is not enough grain data available.
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="columnsToPivot"></param>
        /// <returns></returns>
        public static ForecastingPivotFeaturizerEstimator PivotForecastingData(this TransformsCatalog catalog, string[] columnsToPivot)
        {
            var options = new ForecastingPivotFeaturizerEstimator.Options
            {
                ColumnsToPivot = columnsToPivot
            };

            return new ForecastingPivotFeaturizerEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    public sealed class ForecastingPivotFeaturizerEstimator : IEstimator<ForecastingPivotTransformer>
    {
        private Options _options;

        private readonly IHost _host;

        #region Options
        internal sealed class Options : TransformInputBase
        {

            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of columns to pivot", Name = "ColumnsToPivot", ShortName = "cols", SortOrder = 0)]
            public string[] ColumnsToPivot;
        }

        #endregion

        internal ForecastingPivotFeaturizerEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = Contracts.CheckRef(env, nameof(env)).Register("ShortDropEstimator");
            _host.CheckValue(options.ColumnsToPivot, nameof(options.ColumnsToPivot), "ColumnsToPivot should not be null.");
            _host.CheckNonEmpty(options.ColumnsToPivot, nameof(options.ColumnsToPivot), "Need at least one column.");

            _options = options;
        }

        public ForecastingPivotTransformer Fit(IDataView input)
        {
            return new ForecastingPivotTransformer(_host, _options, input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // A horizon column is always added to the schema.
            // Additionally, the number of the other new columns added is equal to the sum of Dimension[0] for each input column.

            var columns = inputSchema.ToDictionary(x => x.Name);

            columns["Horizon"] = new SchemaShape.Column("Horizon", VectorKind.Scalar, NumberDataViewType.UInt32, false);

            // Make sure all ColumnsToPivot are vectors of type double and the same number of columns.
            // Make new columns based on parsing the input column names.
            foreach (var col in _options.ColumnsToPivot)
            {
                // Make sure the column exists
                var found = inputSchema.TryFindColumn(col, out SchemaShape.Column column);
                if(!found)
                    throw new InvalidOperationException($"Pivot column {col} not found in input");

                var colType = column.ItemType;
                if(column.Kind != VectorKind.Vector)
                    throw new InvalidOperationException($"Pivot column {col} must be a vector");

                if(column.ItemType != NumberDataViewType.Double)
                    throw new InvalidOperationException($"Pivot column {col} must be a vector of type double");

                // By this point the input column should have the correct format.
                // Parse the input column name to figure out if its from rolling window or lag lead.
                // If its from LagLead, new column TODO:
                if (col.Contains("Offsets"))
                {
                    //TODO: Add correct mapping logic here for LagLead
                    columns[""] = new SchemaShape.Column("", VectorKind.Scalar, NumberDataViewType.Double, false);
                }
                else
                {
                    // If its from rolling window, hide original column.
                    columns[col] = new SchemaShape.Column(col, VectorKind.Scalar, NumberDataViewType.Double, false);
                }
            }

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class ForecastingPivotTransformer : ITransformer
    {
        #region Class data members

        internal const string Summary = "Pivots the input colums and drops any rows with N/A";
        internal const string UserName = "ForecastingPivot";
        internal const string ShortName = "fpivot";
        internal const string LoadName = "ForecastingPivot";
        internal const string LoaderSignature = "ForecastingPivot";

        private readonly IHost _host;
        private readonly ForecastingPivotFeaturizerEstimator.Options _options;
        private List<string> _newOutputColumnNames;

        #endregion

        // Normal constructor.
        internal ForecastingPivotTransformer(IHostEnvironment host, ForecastingPivotFeaturizerEstimator.Options options, IDataView input)
        {
            _host = host.Register(nameof(ForecastingPivotTransformer));
            _options = options;

            GenerateOutputColumnNames(true, input);
        }

        // Factory method for SignatureLoadModel.
        internal ForecastingPivotTransformer(IHostEnvironment host, ModelLoadContext ctx)
        {
            _host = host.Register(nameof(ForecastingPivotTransformer));

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array

            var pivotColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < pivotColumns.Length; i++)
                pivotColumns[i] = ctx.Reader.ReadString();

            _options = new ForecastingPivotFeaturizerEstimator.Options
            {
                ColumnsToPivot = pivotColumns
            };

            GenerateOutputColumnNames(false, null);
        }

        // Dont need to validate when transformer is being loaded. Only when its being created from fit.
        // If validate is false, then input will not be checked.
        private void GenerateOutputColumnNames(bool validate, IDataView input)
        {
            _newOutputColumnNames = new List<string>();

            // Only used if validate is true, but needs to be declared.
            ImmutableArray<int>? dimensionsToMatch = default;
            if (validate)
            {
                var firstCol = input.Schema[_options.ColumnsToPivot[0]];
                dimensionsToMatch = (firstCol.Type as VectorDataViewType)?.Dimensions;
            }

            foreach (var col in _options.ColumnsToPivot)
            {
                if (validate)
                {
                    var inputSchema = input.Schema;
                    // Make sure the column exists
                    var column = inputSchema[col];

                    var colType = column.Type as VectorDataViewType;
                    if (colType == null)
                        throw new InvalidOperationException($"Pivot column {col} must be a vector");

                    if (colType.RawType != typeof(VBuffer<double>))
                        throw new InvalidOperationException($"Pivot column {col} must be a vector of type double");

                    if (colType.Dimensions.Length != dimensionsToMatch.Value.Length || colType.Dimensions[1] != dimensionsToMatch.Value[1])
                        throw new InvalidOperationException($"All columns must have the same number of dimensions and the second dimension must be the same size.");
                }

                // By this point the input column should have the correct format.
                // Parse the input column name to figure out if its from rolling window or lag lead.
                if (col.Contains("Offsets"))
                {
                    //TODO: Add correct mapping logic here for LagLead
                    _newOutputColumnNames.Add("");
                }
                else
                {
                    // If its from rolling window, hide original column.
                    _newOutputColumnNames.Add(col);
                }
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            return (IDataTransform)(new ForecastingPivotTransformer(env, ctx).Transform(input));
        }

        public bool IsRowToRowMapper => false;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);
            var schemaBuilder = new DataViewSchema.Builder();
            schemaBuilder.AddColumns(inputSchema.AsEnumerable());

            foreach(var newColName in _newOutputColumnNames)
            {
                schemaBuilder.AddColumn(newColName, NumberDataViewType.Double);
            }

            // Will always add a Horizon columns
            schemaBuilder.AddColumn("Horizon", NumberDataViewType.UInt32);

            return schemaBuilder.ToSchema();
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new InvalidOperationException("Not a RowToRowMapper.");

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FPIVOT T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ForecastingPivotTransformer).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // length of columns array
            // all column names in column array

            ctx.Writer.Write(_options.ColumnsToPivot.Length);
            foreach (var column in _options.ColumnsToPivot)
                ctx.Writer.Write(column);
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        internal ForecastingPivotFeaturizerDataView MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            return new ForecastingPivotFeaturizerDataView(_host, input, _options.ColumnsToPivot, this);
        }
    }

    internal static class ForecastingPivotTransformerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.ForecastingPivot",
            Desc = ForecastingPivotTransformer.Summary,
            UserName = ForecastingPivotTransformer.UserName,
            ShortName = ForecastingPivotTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ShortDrop(IHostEnvironment env, ForecastingPivotFeaturizerEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, ForecastingPivotTransformer.ShortName, input);
            var xf = new ForecastingPivotFeaturizerEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
