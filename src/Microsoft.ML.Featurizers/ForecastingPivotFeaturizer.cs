using System;
using System.Collections.Generic;
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
        public static ForecastingPivotFeaturizerEstimator PivotForcastingData(this TransformsCatalog catalog, string[] columnsToPivot)
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
            // One new column is added per ColumnsToPivot passed in.
            // Rows are also added
            //TODO: Add correct mapping logic here.
            return inputSchema;
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

        #endregion

        // Normal constructor.
        internal ForecastingPivotTransformer(IHostEnvironment host, ForecastingPivotFeaturizerEstimator.Options options, IDataView input)
        {
            _host = host.Register(nameof(ForecastingPivotTransformer));
            _options = options;
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
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            return (IDataTransform)(new ForecastingPivotTransformer(env, ctx).Transform(input));
        }

        public bool IsRowToRowMapper => false;

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            // One new column is added per ColumnsToPivot passed in.
            // Rows are also added
            //TODO: Add correct mapping logic here.
            return inputSchema;
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
        [TlcModule.EntryPoint(Name = "Transforms.ForcastingPivot",
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
