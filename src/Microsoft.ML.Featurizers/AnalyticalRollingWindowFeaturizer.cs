// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
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
using Microsoft.Win32.SafeHandles;
using static Microsoft.ML.Featurizers.CommonExtensions;
using static Microsoft.ML.SchemaShape.Column;

[assembly: LoadableClass(typeof(AnalyticalRollingWindowTransformer), null, typeof(SignatureLoadModel),
    AnalyticalRollingWindowTransformer.UserName, AnalyticalRollingWindowTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(AnalyticalRollingWindowTransformer), null, typeof(SignatureLoadRowMapper),
AnalyticalRollingWindowTransformer.UserName, AnalyticalRollingWindowTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(AnalyticalRollingWindowEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class AnalyticalRollingWindowExtensionClass
    {
        public static AnalyticalRollingWindowEstimator AnalyticalRollingWindowTransformer(this TransformsCatalog catalog, string[] grainColumns, string targetColumn, UInt32 horizon, UInt32 maxWindowSize, UInt32 minWindowSize = 1,
            AnalyticalRollingWindowEstimator.AnalyticalRollingWindowCalculation windowCalculation = AnalyticalRollingWindowEstimator.AnalyticalRollingWindowCalculation.Mean)
        {
            var options = new AnalyticalRollingWindowEstimator.Options {
                GrainColumns = grainColumns,
                TargetColumn = targetColumn,
                Horizon = horizon,
                MaxWindowSize = maxWindowSize,
                MinWindowSize = minWindowSize,
                WindowCalculation = windowCalculation
            };

            return new AnalyticalRollingWindowEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    public class AnalyticalRollingWindowEstimator : IEstimator<AnalyticalRollingWindowTransformer>
    {
        private Options _options;
        private readonly IHost _host;

        #region Options

        internal sealed class Options: TransformInputBase
        {
            [Argument((ArgumentType.MultipleUnique | ArgumentType.Required), HelpText = "List of grain columns",
                Name = "GrainColumns", ShortName = "grains", SortOrder = 0)]
            public string[] GrainColumns;

            [Argument(ArgumentType.Required, HelpText = "Target column",
                Name = "TargetColumn", ShortName = "target", SortOrder = 1)]
            public string TargetColumn;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Maximum horizon value",
                Name = "Horizon", ShortName = "hor", SortOrder = 2)]
            public UInt32 Horizon;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Maximum window size",
                Name = "MaxWindowSize", ShortName = "maxsize", SortOrder = 3)]
            public UInt32 MaxWindowSize;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Minimum window size",
                Name = "MinWindowSize", ShortName = "minsize", SortOrder = 4)]
            public UInt32 MinWindowSize = 1;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "What window calculation to use",
                Name = "WindowCalculation", ShortName = "calc", SortOrder = 5)]
            public AnalyticalRollingWindowCalculation WindowCalculation = AnalyticalRollingWindowCalculation.Mean;
        }

        #endregion

        #region Class Enums

        public enum AnalyticalRollingWindowCalculation : byte {
            Mean = 1
        };

        #endregion

        internal AnalyticalRollingWindowEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(AnalyticalRollingWindowEstimator));
            Contracts.CheckNonEmpty(options.GrainColumns, nameof(options.GrainColumns));

            _options = options;
        }

        public AnalyticalRollingWindowTransformer Fit(IDataView input)
        {
            return new AnalyticalRollingWindowTransformer(_host, input, _options);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            var inputColumn = columns[_options.TargetColumn];

            if (!AnalyticalRollingWindowTransformer.TypedColumn.IsColumnTypeSupported(inputColumn.ItemType.RawType))
                throw new InvalidOperationException($"Type {inputColumn.ItemType.RawType.ToString()} for column {_options.TargetColumn} not a supported type.");

            var columnName = $"{_options.TargetColumn}_{Enum.GetName(typeof(AnalyticalRollingWindowCalculation), _options.WindowCalculation)}_Hor{_options.Horizon}_MinWin{_options.MinWindowSize}_MaxWin{_options.MaxWindowSize}";

            columns[_options.TargetColumn] = new SchemaShape.Column(columnName, VectorKind.Vector,
                NumberDataViewType.Double, false, inputColumn.Annotations);

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class AnalyticalRollingWindowTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "Performs an analaytical calculation over a rolling timeseries window";
        internal const string UserName = "AnalyticalRollingWindow";
        internal const string ShortName = "AnalyticRollingWindow";
        internal const string LoaderSignature = "AnalyticalRollingWindow";

        private TypedColumn _column;
        private AnalyticalRollingWindowEstimator.Options _options;

        #endregion

        internal AnalyticalRollingWindowTransformer(IHostEnvironment host, IDataView input, AnalyticalRollingWindowEstimator.Options options) :
            base(host.Register(nameof(AnalyticalRollingWindowTransformer)))
        {
            var schema = input.Schema;
            _options = options;

            _column = TypedColumn.CreateTypedColumn(_options.TargetColumn, schema[_options.TargetColumn].Type.RawType.ToString(), _options);

            _column.CreateTransformerFromEstimator(input);
        }

        // Factory method for SignatureLoadModel.
        internal AnalyticalRollingWindowTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(AnalyticalRollingWindowTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int length of grainColumns
            // string[] grainColumns
            // string targetColumn
            // uint32 horizon
            // uint32 maxWindowSize
            // uint32 minWindowSize
            // byte windowCalculation
            // string columnType
            // int length of c++ byte array
            // byte array from c++

            var grainColumns = new string[ctx.Reader.ReadInt32()];
            for(int i = 0; i < grainColumns.Length; i++)
            {
                grainColumns[i] = ctx.Reader.ReadString();
            }

            var targetColumn = ctx.Reader.ReadString();
            var horizon = ctx.Reader.ReadUInt32();
            var maxWindowSize = ctx.Reader.ReadUInt32();
            var minWindowSize = ctx.Reader.ReadUInt32();
            var windowCalculation = ctx.Reader.ReadByte();

            _options = new AnalyticalRollingWindowEstimator.Options()
            {
                GrainColumns = grainColumns,
                TargetColumn = targetColumn,
                Horizon = horizon,
                MaxWindowSize = maxWindowSize,
                MinWindowSize = minWindowSize,
                WindowCalculation = (AnalyticalRollingWindowEstimator.AnalyticalRollingWindowCalculation)windowCalculation
            };

            _column = TypedColumn.CreateTypedColumn(targetColumn, ctx.Reader.ReadString(), _options);

            // Load the C++ state and create the C++ transformer.
            var dataLength = ctx.Reader.ReadInt32();
            var data = ctx.Reader.ReadByteArray(dataLength);
            _column.CreateTransformerFromSavedData(data);
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new AnalyticalRollingWindowTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "ANROLW T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(AnalyticalRollingWindowTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int length of grainColumns
            // string[] grainColumns
            // string targetColumn
            // uint32 horizon
            // uint32 maxWindowSize
            // uint32 minWindowSize
            // byte windowCalculation
            // string columnType
            // int length of c++ byte array
            // byte array from c++

            ctx.Writer.Write(_options.GrainColumns.Length);
            foreach (var grain in _options.GrainColumns)
            {
                ctx.Writer.Write(grain);
            }
            ctx.Writer.Write(_options.TargetColumn);
            ctx.Writer.Write(_options.Horizon);
            ctx.Writer.Write(_options.MaxWindowSize);
            ctx.Writer.Write(_options.MinWindowSize);
            ctx.Writer.Write((byte)_options.WindowCalculation);
            ctx.Writer.Write(_column.Type);
            // Save native state.
            var data = _column.CreateTransformerSaveData();
            ctx.Writer.Write(data.Length);
            ctx.Writer.Write(data);
        }

        public void Dispose()
        {

            _column.Dispose();

        }

        #region Native Safe handle classes
        internal delegate bool DestroyTransformedVectorDataNative(IntPtr handle, IntPtr itemSize, out IntPtr errorHandle);
        internal class TransformedVectorDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
        {
            private readonly DestroyTransformedVectorDataNative _destroyTransformedDataHandler;
            private readonly IntPtr _itemSize;

            public TransformedVectorDataSafeHandle(IntPtr handle, IntPtr itemSize, DestroyTransformedVectorDataNative destroyTransformedDataHandler) : base(true)
            {
                SetHandle(handle);
                _destroyTransformedDataHandler = destroyTransformedDataHandler;
                _itemSize = itemSize;
            }

            protected override bool ReleaseHandle()
            {
                // Not sure what to do with error stuff here.  There shouldn't ever be one though.
                var success = _destroyTransformedDataHandler(handle, _itemSize, out IntPtr errorHandle);
                return success;
            }
        }

        #endregion

        #region ColumnInfo

        #region BaseClass

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Source;
            internal readonly string Type;

            private protected TransformerEstimatorSafeHandle TransformerHandler;
            private static readonly Type[] _supportedTypes = new Type[] { typeof(double) };

            private protected string[] GrainColumns;

            internal TypedColumn(string source, string type, string[] grainColumns)
            {
                Source = source;
                Type = type;
                GrainColumns = grainColumns;
            }

            internal abstract void CreateTransformerFromEstimator(IDataView input);
            private protected abstract unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize);
            private protected abstract bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
            private protected abstract bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected abstract bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            public abstract void Dispose();

            public abstract Type ReturnType();

            internal byte[] CreateTransformerSaveData()
            {

                var success = CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var savedDataHandle = new SaveDataSafeHandle(buffer, bufferSize))
                {
                    byte[] savedData = new byte[bufferSize.ToInt32()];
                    Marshal.Copy(buffer, savedData, 0, savedData.Length);
                    return savedData;
                }
            }

            internal unsafe void CreateTransformerFromSavedData(byte[] data)
            {
                fixed (byte* rawData = data)
                {
                    IntPtr dataSize = new IntPtr(data.Count());
                    CreateTransformerFromSavedDataHelper(rawData, dataSize);
                }
            }

            internal static bool IsColumnTypeSupported(Type type)
            {
                return _supportedTypes.Contains(type);
            }

            internal static TypedColumn CreateTypedColumn(string source, string type, AnalyticalRollingWindowEstimator.Options options)
            {
                if (type == typeof(double).ToString())
                {
                    return new DoubleTypedColumn(source, options);
                }

                throw new InvalidOperationException($"Column {source} has an unsupported type {type}.");
            }
        }

        internal abstract class TypedColumn<TSourceType, TOutputType> : TypedColumn
        {
            private protected IEnumerator<ReadOnlyMemory<char>>[] GrainEnumerators;
            private protected readonly AnalyticalRollingWindowEstimator.Options Options;

            internal TypedColumn(string source, string type, AnalyticalRollingWindowEstimator.Options options) :
                base(source, type, options.GrainColumns)
            {
                Options = options;

                // Initialize to the correct length
                GrainEnumerators = new IEnumerator<ReadOnlyMemory<char>>[GrainColumns.Length];
            }

            internal abstract TOutputType Transform(IntPtr grainsArray, IntPtr grainsArraySize, TSourceType input);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected unsafe abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, TSourceType value, out FitResult fitResult, out IntPtr errorHandle);
            private protected abstract bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);

            private protected TransformerEstimatorSafeHandle CreateTransformerFromEstimatorBase(IDataView input)
            {
                var success = CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using (var estimatorHandle = new TransformerEstimatorSafeHandle(estimator, DestroyEstimatorHelper))
                {
                    TrainingState trainingState;
                    FitResult fitResult;

                    InitializeGrainEnumerators(input);

                    // Can't use a using with this because it potentially needs to be reset. Manually disposing as needed.
                    var data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                    data.MoveNext();
                    while (true)
                    {
                        // Get the state of the native estimator.
                        success = GetStateHelper(estimatorHandle, out trainingState, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        // If we are no longer training then exit loop.
                        if (trainingState != TrainingState.Training)
                            break;

                        // Build the string array
                        GCHandle[] grainHandles = default;
                        GCHandle arrayHandle = default;
                        try
                        {
                            grainHandles = new GCHandle[GrainColumns.Length];
                            IntPtr[] grainArray = new IntPtr[grainHandles.Length];
                            for (int grainIndex = 0; grainIndex < grainHandles.Length; grainIndex++)
                            {
                                grainHandles[grainIndex] = GCHandle.Alloc(Encoding.UTF8.GetBytes(GrainEnumerators[grainIndex].Current.ToString() + char.MinValue), GCHandleType.Pinned);
                                grainArray[grainIndex] = grainHandles[grainIndex].AddrOfPinnedObject();
                            }

                            // Fit the estimator
                            arrayHandle = GCHandle.Alloc(grainArray, GCHandleType.Pinned);
                            success = FitHelper(estimatorHandle, arrayHandle.AddrOfPinnedObject(), new IntPtr(grainArray.Length), data.Current, out fitResult, out errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                        }
                        finally
                        {
                            arrayHandle.Free();
                            foreach (var handle in grainHandles)
                            {
                                handle.Free();
                            }
                        }

                        // If we need to reset the data to the beginning.
                        if (fitResult == FitResult.ResetAndContinue)
                        {
                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                        }

                        // If we are at the end of the data.
                        if (!data.MoveNext())
                        {
                            OnDataCompletedHelper(estimatorHandle, out errorHandle);
                            if (!success)
                                throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                            // Re-initialize the data
                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                            data.MoveNext();

                            InitializeGrainEnumerators(input);
                        }
                    }

                    // When done training complete the estimator.
                    success = CompleteTrainingHelper(estimatorHandle, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // Create the native transformer from the estimator;
                    success = CreateTransformerFromEstimatorHelper(estimatorHandle, out IntPtr transformer, out errorHandle);
                    if (!success)
                        throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                    // Manually dispose of the IEnumerator since we dont have a using statement;
                    data.Dispose();
                    for (int i = 0; i < GrainColumns.Length; i++)
                    {
                        // Manually dispose because we can't use a using statement.
                        if (GrainEnumerators[i] != null)
                            GrainEnumerators[i].Dispose();
                    }

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }

            private void InitializeGrainEnumerators(IDataView input)
            {
                // Create enumerators for the grain columns. Cant use using because it may need to be reset.
                for (int i = 0; i < GrainColumns.Length; i++)
                {
                    // Manually dispose because we can't use a using statement.
                    if (GrainEnumerators[i] != null)
                        GrainEnumerators[i].Dispose();

                    // Inititialize the enumerator and move it to a valid position.
                    GrainEnumerators[i] = input.GetColumn<ReadOnlyMemory<char>>(GrainColumns[i]).GetEnumerator();
                    GrainEnumerators[i].MoveNext();
                }
            }

            public override Type ReturnType()
            {
                return typeof(TOutputType);
            }
        }

        #endregion

        #region DoubleTypedColumn

        internal sealed class DoubleTypedColumn : TypedColumn<double, VBuffer<double>>
        {
            internal DoubleTypedColumn(string source, AnalyticalRollingWindowEstimator.Options options) :
                base(source, typeof(double).ToString(), options)
            {
            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateEstimatorNative(AnalyticalRollingWindowEstimator.AnalyticalRollingWindowCalculation windowCalculation, UInt32 horizon, UInt32 maxWindowSize, UInt32 minWindowSize, out IntPtr estimator, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
            internal override void CreateTransformerFromEstimator(IDataView input)
            {
                TransformerHandler = CreateTransformerFromEstimatorBase(input);
            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
            private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
            {
                var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
                if (!result)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, IntPtr grainsArray, IntPtr grainsArraySize, double value, out double* output, out IntPtr outputSize, out IntPtr errorHandle);
            internal unsafe override VBuffer<double> Transform(IntPtr grainsArray, IntPtr grainsArraySize, double input)
            {
                var success = TransformDataNative(TransformerHandler, grainsArray, grainsArraySize, input, out double* output, out IntPtr outputSize, out IntPtr errorHandle);
                if (!success)
                    throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

                using var handler = new TransformedVectorDataSafeHandle(new IntPtr(output), outputSize, DestroyTransformedDataNative);

                var outputArray = new double[outputSize.ToInt32()];

                for(int i = 0; i < outputSize.ToInt32(); i++)
                {
                    outputArray[i] = *output++;
                }

                var buffer = new VBuffer<double>(outputSize.ToInt32(), outputArray);
                return buffer;
            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_DestroyTransformedData"), SuppressUnmanagedCodeSecurity]
            private static unsafe extern bool DestroyTransformedDataNative(IntPtr items, IntPtr itemsSize, out IntPtr errorHandle);

            private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
            {
                return CreateEstimatorNative(Options.WindowCalculation, Options.Horizon, Options.MaxWindowSize, Options.MinWindowSize, out estimator, out errorHandle);
            }

            private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
                CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

            private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
                DestroyEstimatorNative(estimator, out errorHandle);

            private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
                DestroyTransformerNative(transformer, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool FitNative(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, double value, out FitResult fitResult, out IntPtr errorHandle);
            private protected override bool FitHelper(TransformerEstimatorSafeHandle estimator, IntPtr grainsArray, IntPtr grainsArraySize, double value, out FitResult fitResult, out IntPtr errorHandle)
            {
                return FitNative(estimator, grainsArray, grainsArraySize, value, out fitResult, out errorHandle);

            }

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    CompleteTrainingNative(estimator, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
            private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
                CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
            private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
                GetStateNative(estimator, out trainingState, out errorHandle);

            [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
            private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
            private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
                    OnDataCompletedNative(estimator, out errorHandle);

            public override void Dispose()
            {
                if (!TransformerHandler.IsClosed)
                    TransformerHandler.Dispose();
            }
        }

        #endregion

        #endregion

        private sealed class Mapper : MapperBase
        {
            #region Class members

            private readonly AnalyticalRollingWindowTransformer _parent;
            private readonly string _outputColumnName;

            #endregion

            public Mapper(AnalyticalRollingWindowTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _outputColumnName = $"{_parent._options.TargetColumn}_{Enum.GetName(typeof(AnalyticalRollingWindowEstimator.AnalyticalRollingWindowCalculation), _parent._options.WindowCalculation)}_Hor{_parent._options.Horizon}_MinWin{_parent._options.MinWindowSize}_MaxWin{_parent._options.MaxWindowSize}";
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                // To add future support for when this will do multiple columns at once, output will be a 2d vector so nothing will need to change when that is implemented.
                return new[] { new DataViewSchema.DetachedColumn(_outputColumnName, new VectorDataViewType(NumberDataViewType.Double, 1, (int)_parent._options.Horizon)) };
            }

            private Delegate MakeGetter<TSourceType, TOutputType>(DataViewRow input, int iinfo)
            {
                var inputColumn = input.Schema[_parent._column.Source];
                var srcGetterScalar = input.GetGetter<TSourceType>(inputColumn);

                // Initialize grain getters.
                int grainColumnCount = _parent._options.GrainColumns.Length;
                ValueGetter<ReadOnlyMemory<char>>[] grainGetters = new ValueGetter<ReadOnlyMemory<char>>[grainColumnCount];
                for (int i = 0; i < grainGetters.Length; i++)
                {
                    grainGetters[i] = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent._options.GrainColumns[i]]);
                }

                // Declaring these outside so they are only done once
                GCHandle[] grainHandles = default;
                GCHandle arrayHandle = default;

                ValueGetter<TOutputType> result = (ref TOutputType dst) =>
                {
                    ReadOnlyMemory<char> grainValue = default;
                    TSourceType value = default;

                    // Build the string array
                    try
                    {
                        grainHandles = new GCHandle[grainColumnCount];
                        IntPtr[] grainArray = new IntPtr[grainHandles.Length];
                        for (int grainIndex = 0; grainIndex < grainHandles.Length; grainIndex++)
                        {
                            grainGetters[grainIndex](ref grainValue);
                            grainHandles[grainIndex] = GCHandle.Alloc(Encoding.UTF8.GetBytes(grainValue.ToString() + char.MinValue), GCHandleType.Pinned);
                            grainArray[grainIndex] = grainHandles[grainIndex].AddrOfPinnedObject();
                        }

                        srcGetterScalar(ref value);

                        arrayHandle = GCHandle.Alloc(grainArray, GCHandleType.Pinned);
                        dst = ((TypedColumn<TSourceType, TOutputType>)_parent._column).Transform(arrayHandle.AddrOfPinnedObject(), new IntPtr(grainArray.Length), value);
                    }
                    finally
                    {
                        arrayHandle.Free();
                        foreach (var handle in grainHandles)
                        {
                            handle.Free();
                        }
                    }
                };

                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Type inputType = input.Schema[_parent._column.Source].Type.RawType;
                Type outputType = _parent._column.ReturnType();

                return Utils.MarshalInvoke(MakeGetter<int, int>, new Type[] { inputType, outputType }, input, iinfo);
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var active = new bool[InputSchema.Count];
                for (int i = 0; i < InputSchema.Count; i++)
                {
                    if (_parent._options.GrainColumns.Any(x => x == InputSchema[i].Name) || _parent._options.TargetColumn == InputSchema[i].Name)
                    {
                        active[i] = true;
                    }
                }

                return col => active[col];
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    internal static class AnalyticalRollingWindowEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.AnalyticalRollingWindow",
            Desc = AnalyticalRollingWindowTransformer.Summary,
            UserName = AnalyticalRollingWindowTransformer.UserName,
            ShortName = AnalyticalRollingWindowTransformer.ShortName)]
        public static CommonOutputs.TransformOutput AnalyticalRollingWindow(IHostEnvironment env, AnalyticalRollingWindowEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, AnalyticalRollingWindowTransformer.ShortName, input);
            var xf = new AnalyticalRollingWindowEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
