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

[assembly: LoadableClass(typeof(ShortDropTransformer), null, typeof(SignatureLoadModel),
    ShortDropTransformer.UserName, ShortDropTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IDataTransform), typeof(ShortDropTransformer), null, typeof(SignatureLoadDataTransform),
   ShortDropTransformer.UserName, ShortDropTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(TimeSeriesTransformerEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class ShortGrainDropperExtensionClass
    {

        /// <summary>
        /// Drops rows when there is not enough grain data available.
        /// </summary>
        /// <param name="catalog"></param>
        /// <param name="grainColumns"></param>
        /// <param name="horizon"></param>
        /// <param name="maxWindowSize"></param>
        /// <param name="offsets"></param>
        /// <param name="crossValidations"></param>
        /// <returns></returns>
        public static ShortGrainDropperEstimator DropShortGrains(this TransformsCatalog catalog, string[] grainColumns, UInt32 horizon, UInt32 maxWindowSize, long[] offsets, UInt32 crossValidations = 0)
        {
            var options = new ShortGrainDropperEstimator.Options
            {
                GrainColumns = grainColumns,
                Horizon = horizon,
                MaxWindowSize = maxWindowSize,
                Offsets = offsets,
                CrossValidations = crossValidations
            };

            return new ShortGrainDropperEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    public sealed class ShortGrainDropperEstimator : IEstimator<ShortDropTransformer>
    {
        private Options _options;

        private readonly IHost _host;

        #region Options
        internal sealed class Options : TransformInputBase
        {

            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of grain columns", Name = "GrainColumns", ShortName = "grains", SortOrder = 0)]
            public string[] GrainColumns;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Maximum horizon value",
                Name = "Horizon", ShortName = "hor", SortOrder = 1)]
            public UInt32 Horizon;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Maximum window size",
                Name = "MaxWindowSize", ShortName = "maxsize", SortOrder = 2)]
            public UInt32 MaxWindowSize;

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Lag and Lead offset to use. A negative number is a lag, positive is a lead",
                Name = "offsets", ShortName = "off", SortOrder = 3)]
            public long[] Offsets;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of cross validations being performed.",
                Name = "CrossValidations", ShortName = "crossv", SortOrder = 2)]
            public UInt32 CrossValidations;
        }

        #endregion

        internal ShortGrainDropperEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = Contracts.CheckRef(env, nameof(env)).Register("ShortDropEstimator");
            _host.CheckValue(options.GrainColumns, nameof(options.GrainColumns), "Grain columns should not be null.");
            _host.CheckNonEmpty(options.GrainColumns, nameof(options.GrainColumns), "Need at least one grain column.");

            _options = options;
        }

        public ShortDropTransformer Fit(IDataView input)
        {
            return new ShortDropTransformer(_host, _options, input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            // We dont change the schema, we just drop rows
            return inputSchema;
        }
    }

    public sealed class ShortDropTransformer : ITransformer, IDisposable
    {
        #region Class data members

        internal const string Summary = "Drops rows if there aren't enough values per grain.";
        internal const string UserName = "ShortDrop";
        internal const string ShortName = "sgd";
        internal const string LoadName = "ShortDrop";
        internal const string LoaderSignature = "ShortDrop";

        private readonly IHost _host;
        private readonly ShortGrainDropperEstimator.Options _options;
        internal TransformerEstimatorSafeHandle TransformerHandle;

        #endregion

        // Normal constructor.
        internal ShortDropTransformer(IHostEnvironment host, ShortGrainDropperEstimator.Options options, IDataView input)
        {
            _host = host.Register(nameof(ShortDropTransformer));
            _options = options;

            TransformerHandle = CreateTransformerFromEstimator(input);
        }

        // Factory method for SignatureLoadModel.
        internal ShortDropTransformer(IHostEnvironment host, ModelLoadContext ctx)
        {
            _host = host.Register(nameof(ShortDropTransformer));

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array
            // Horizon
            // MaxWindowSize
            // length of offset array
            // offsets
            // CrossValidation
            // length of C++ state array
            // C++ byte state array

            var grainColumns = new string[ctx.Reader.ReadInt32()];
            for (int i = 0; i < grainColumns.Length; i++)
                grainColumns[i] = ctx.Reader.ReadString();

            var horizon = ctx.Reader.ReadUInt32();
            var maxWindow = ctx.Reader.ReadUInt32();

            var offsets = new long[ctx.Reader.ReadInt32()];
            for (int i = 0; i < offsets.Length; i++)
                offsets[i] = ctx.Reader.ReadInt64();

            var crossValidation = ctx.Reader.ReadUInt32();

            _options = new ShortGrainDropperEstimator.Options
            {
                GrainColumns = grainColumns,
                Horizon = horizon,
                MaxWindowSize = maxWindow,
                Offsets = offsets
            };

            var nativeState = ctx.Reader.ReadByteArray();
            TransformerHandle = CreateTransformerFromSavedData(nativeState);
        }

        private unsafe TransformerEstimatorSafeHandle CreateTransformerFromSavedData(byte[] nativeState)
        {
            fixed (byte* rawStatePointer = nativeState)
            {
                IntPtr dataSize = new IntPtr(nativeState.Count());
                var result = CreateTransformerFromSavedDataNative(rawStatePointer, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            return (IDataTransform)(new ShortDropTransformer(env, ctx).Transform(input));
        }

        private unsafe TransformerEstimatorSafeHandle CreateTransformerFromEstimator(IDataView input)
        {
            IntPtr estimator;
            IntPtr errorHandle;
            bool success;

            var grainColumns = input.Schema.Where(x => _options.GrainColumns.Contains(x.Name)).Select(x => TypedColumn.CreateTypedColumn(x)).ToDictionary(x => x.Column.Name);

            fixed (long* offsets = _options.Offsets)
            fixed (UInt32* crossVal = &_options.CrossValidations)
            {
                success = CreateEstimatorNative(_options.MaxWindowSize, offsets, new IntPtr(_options.Offsets.Length), _options.Horizon, _options.CrossValidations > 0 ? crossVal : null , out estimator, out errorHandle);
            }
            if (!success)
                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

            using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorNative))
            {
                TrainingState trainingState;
                FitResult fitResult;

                // Create buffer to hold binary data
                var memoryStream = new MemoryStream(4096);
                var binaryWriter = new BinaryWriter(memoryStream, Encoding.UTF8);

                // Can't use a using with this because it potentially needs to be reset. Manually disposing as needed.
                var cursor = input.GetRowCursorForAllColumns();
                // Initialize getters
                foreach (var column in grainColumns.Values)
                    column.InitializeGetter(cursor);

                // Start the loop with the cursor in a valid state already.
                cursor.MoveNext();
                while (true)
                {
                    // Get the state of the native estimator.
                    success = GetStateNative(estimatorHandle, out trainingState, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // If we are no longer training then exit loop.
                    if (trainingState != TrainingState.Training)
                        break;

                    // Build byte array to send column data to native featurizer
                    BuildColumnByteArray(grainColumns, ref binaryWriter);

                    // Fit the estimator
                    fixed (byte* bufferPointer = memoryStream.GetBuffer())
                    {
                        var binaryArchiveData = new NativeBinaryArchiveData() { Data = bufferPointer, DataSize = new IntPtr(memoryStream.Position) };
                        success = FitNative(estimatorHandle, binaryArchiveData, out fitResult, out errorHandle);
                    }

                    // Reset memory stream to 0
                    memoryStream.Position = 0;

                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // If we need to reset the data to the beginning.
                    if (fitResult == FitResult.ResetAndContinue)
                        ResetCursor(input, ref cursor, grainColumns);

                    // If we are at the end of the data.
                    if (!cursor.MoveNext())
                    {
                        OnDataCompletedNative(estimatorHandle, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        ResetCursor(input, ref cursor, grainColumns);
                    }
                }

                // When done training complete the estimator.
                success = CompleteTrainingNative(estimatorHandle, out errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                // Create the native transformer from the estimator;
                success = CreateTransformerFromEstimatorNative(estimatorHandle, out IntPtr transformer, out errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                // Manually dispose of the IEnumerator since we dont have a using statement;
                cursor.Dispose();

                return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }
        }

        private void ResetCursor(IDataView input, ref DataViewRowCursor cursor, Dictionary<string, TypedColumn> columns)
        {
            cursor.Dispose();
            cursor = input.GetRowCursorForAllColumns();

            // Initialize getters
            foreach (var column in columns.Values)
                column.InitializeGetter(cursor);

            // Move cursor to valid position
            cursor.MoveNext();
        }

        private void BuildColumnByteArray(Dictionary<string, TypedColumn> allColumns, ref BinaryWriter binaryWriter)
        {
            foreach (var column in _options.GrainColumns)
            {
                allColumns[column].SerializeValue(ref binaryWriter);
            }
        }

        public bool IsRowToRowMapper => false;

        // Schema not changed
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            return inputSchema;
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) => throw new InvalidOperationException("Not a RowToRowMapper.");

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SGDROP T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ShortDropTransformer).Assembly.FullName);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // length of grain column array
            // all column names in grain column array
            // Horizon
            // MaxWindowSize
            // length of offset array
            // offsets
            // CrossValidation
            // length of C++ state array
            // C++ byte state array

            ctx.Writer.Write(_options.GrainColumns.Length);
            foreach (var column in _options.GrainColumns)
                ctx.Writer.Write(column);

            ctx.Writer.Write(_options.Horizon);
            ctx.Writer.Write(_options.MaxWindowSize);
            ctx.Writer.Write(_options.Offsets.Length);

            foreach (var offset in _options.Offsets)
                ctx.Writer.Write(offset);

            ctx.Writer.Write(_options.CrossValidations);

            var data = CreateTransformerSaveData();
            ctx.Writer.Write(data.Length);
            ctx.Writer.Write(data);
        }

        private byte[] CreateTransformerSaveData()
        {
            var success = CreateTransformerSaveDataNative(TransformerHandle, out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            if (!success)
                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

            using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
            {
                byte[] savedData = new byte[bufferSize.ToInt32()];
                Marshal.Copy(buffer, savedData, 0, savedData.Length);
                return savedData;
            }
        }

        public IDataView Transform(IDataView input) => MakeDataTransform(input);

        internal ShortGrainDropperDataView MakeDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            return new ShortGrainDropperDataView(_host, input, _options.GrainColumns, this);
        }

        internal TransformerEstimatorSafeHandle CloneTransformer() => CreateTransformerFromSavedData(CreateTransformerSaveData());

        public void Dispose()
        {
            if (!TransformerHandle.IsClosed)
                TransformerHandle.Close();
        }

        #region C++ function declarations
        // TODO: Update entry points

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool CreateEstimatorNative(UInt32 windowSize, long* offsets, IntPtr offsetsSize, UInt32 horizon, UInt32* crossValidations, out IntPtr estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, NativeBinaryArchiveData data, out FitResult fitResult, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_CreateTransformerFromSavedData"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_OnDataCompleted"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

        [DllImport("Featurizers", EntryPoint = "ShortGrainDropperDropperFeaturizer_GetState"), SuppressUnmanagedCodeSecurity]
        private static unsafe extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);

        #endregion

        #region Typed Columns

        private abstract class TypedColumn
        {
            internal readonly DataViewSchema.Column Column;
            internal TypedColumn(DataViewSchema.Column column)
            {
                Column = column;
            }

            internal abstract void InitializeGetter(DataViewRowCursor cursor);
            internal abstract void SerializeValue(ref BinaryWriter binaryWriter);
            internal abstract TypeId GetTypeId();

            internal static TypedColumn CreateTypedColumn(DataViewSchema.Column column)
            {
                var type = column.Type.RawType.ToString();
                if (type == typeof(ReadOnlyMemory<char>).ToString())
                    return new StringTypedColumn(column);

                throw new InvalidOperationException($"Unsupported type {type}");
            }
        }

        private abstract class TypedColumn<T> : TypedColumn
        {
            private ValueGetter<T> _getter;
            private T _value;

            internal TypedColumn(DataViewSchema.Column column) :
                base(column)
            {
                _value = default;
            }

            internal override void InitializeGetter(DataViewRowCursor cursor)
            {
                _getter = cursor.GetGetter<T>(Column);
            }

            internal T GetValue()
            {
                _getter(ref _value);
                return _value;
            }

            internal override TypeId GetTypeId()
            {
                return typeof(T).GetNativeTypeIdFromType();
            }
        }

        private class StringTypedColumn : TypedColumn<ReadOnlyMemory<char>>
        {

            internal StringTypedColumn(DataViewSchema.Column column) :
                base(column)
            {
            }

            internal override void SerializeValue(ref BinaryWriter binaryWriter)
            {
                var value = GetValue().ToString();
                var stringBytes = Encoding.UTF8.GetBytes(value);

                binaryWriter.Write(stringBytes.Length);

                binaryWriter.Write(stringBytes);
            }
        }

        private class DateTimeTypedColumn : TypedColumn<DateTime>
        {
            private static readonly DateTime _unixEpoch = new DateTime(1970, 1, 1);
            private readonly bool _isNullable;

            internal DateTimeTypedColumn(DataViewSchema.Column column, bool isNullable = false) :
                base(column)
            {
                _isNullable = isNullable;
            }

            internal override void SerializeValue(ref BinaryWriter binaryWriter)
            {
                var dateTime = GetValue();

                var value = dateTime.Subtract(_unixEpoch).Ticks / TimeSpan.TicksPerSecond;

                if (_isNullable)
                    binaryWriter.Write(true);

                binaryWriter.Write(value);
            }
        }

        #endregion
    }

    internal static class ShortDropTransformerEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.ShortDrop",
            Desc = ShortDropTransformer.Summary,
            UserName = ShortDropTransformer.UserName,
            ShortName = ShortDropTransformer.ShortName)]
        public static CommonOutputs.TransformOutput ShortDrop(IHostEnvironment env, ShortGrainDropperEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, ShortDropTransformer.ShortName, input);
            var xf = new ShortGrainDropperEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
