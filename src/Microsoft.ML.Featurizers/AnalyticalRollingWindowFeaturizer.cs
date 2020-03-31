// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
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
        public static AnalyticalRollingWindowEstimator AnalyticalRollingWindowTransformer(this TransformsCatalog catalog, string[] grainColumns, string targetColumn, UInt32 horizon, UInt32 maxWindowSize, UInt32 minWindowSize,
            AnalyticalRollingWindowEstimator.AnalyticalRollingWindowCalculation windowCalculation = AnalyticalRollingWindowEstimator.AnalyticalRollingWindowCalculation.Mean, string inputColumnName = null)
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
            public UInt32 MinWindowSize;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "What window calculation to use",
                Name = "WindowCalculation", ShortName = "calc", SortOrder = 5)]
            public AnalyticalRollingWindowCalculation WindowCalculation;
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

            columns[_options.TargetColumn] = new SchemaShape.Column(_options.TargetColumn + "_" + Enum.GetName(typeof(AnalyticalRollingWindowCalculation), _options.WindowCalculation), VectorKind.Vector,
                NumberDataViewType.Double, false, inputColumn.Annotations);

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class AnalyticalRollingWindowTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "Performs an analaytical calculation over a rolling timeseries window";
        internal const string UserName = "AnalyticalRollingWindowTransformer";
        internal const string ShortName = "AnalyticalRollingWindowTransformer";
        internal const string LoadName = "AnalyticalRollingWindowTransformer";
        internal const string LoaderSignature = "AnalyticalRollingWindowTransformer";

        private TypedColumn _column;
        private AnalyticalRollingWindowEstimator.Options _options;

        #endregion

        internal AnalyticalRollingWindowTransformer(IHostEnvironment host, IDataView input, AnalyticalRollingWindowEstimator.Options options) :
            base(host.Register(nameof(AnalyticalRollingWindowTransformer)))
        {
            var schema = input.Schema;
            _options = options;

            _column = TypedColumn.CreateTypedColumn(_options.TargetColumn + "RollingWindow", _options.TargetColumn, schema[_options.TargetColumn].Type.RawType.ToString());
            // TODO: wrapper
            //_column.CreateTransformerFromEstimator(input);
        }

        // Factory method for SignatureLoadModel.
        internal AnalyticalRollingWindowTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(AnalyticalRollingWindowTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            /* Codegen: Edit this format as needed */
            // *** Binary format ***
            // int number of column pairs
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            _options = new AnalyticalRollingWindowEstimator.Options();

            /* Codegen: Load any additional Options members here */

            // TODO: wrapper
            //_columns = new TypedColumn[columnCount];
            //for (int i = 0; i < columnCount; i++)
            //{
            //    _columns[i] = TypedColumn.CreateTypedColumn(ctx.Reader.ReadString(), ctx.Reader.ReadString(), ctx.Reader.ReadString());

            //    // Load the C++ state and create the C++ transformer.
            //    var dataLength = ctx.Reader.ReadInt32();
            //    var data = ctx.Reader.ReadByteArray(dataLength);
            //    _columns[i].CreateTransformerFromSavedData(data);
            //}
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new AnalyticalRollingWindowTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            /* Codegen: Change these as needed */
            return new VersionInfo(
                modelSignature: "ANROLW T",
                verWrittenCur: 0x00010001, /* Codegen: Update version numbers as necessary */
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

            /* Codegen: Edit this format as needed */
            // *** Binary format ***
            // for each column pair:
            //      string output column  name
            //      string input column name
            //      column type
            //      int length of c++ byte array
            //      byte array from c++

            /* Codegen: Write any _options members needed here */

            // TODO: wrapper
            //foreach (var column in _columns)
            //{
            //    ctx.Writer.Write(column.Name);
            //    ctx.Writer.Write(column.Source);
            //    ctx.Writer.Write(column.Type.ToString());

            //    // Save C++ state
            //    var data = column.CreateTransformerSaveData();
            //    ctx.Writer.Write(data.Length);
            //    ctx.Writer.Write(data);
            //}
        }

        public void Dispose()
        {

            // TODO: wrapper
            //_column.Dispose();

        }

        #region ColumnInfo

        #region BaseClass

        internal abstract class TypedColumn : IDisposable
        {
            internal readonly string Name;
            internal readonly string Source;
            internal readonly string Type;

            private protected TransformerEstimatorSafeHandle TransformerHandler;
            private static readonly Type[] _supportedTypes = new Type[] { typeof(sbyte), typeof(short), typeof(int), typeof(long), typeof(byte), typeof(ushort), typeof(uint), typeof(ulong), typeof(float), typeof(double) };

            internal TypedColumn(string name, string source, string type)
            {
                Name = name;
                Source = source;
                Type = type;
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

            internal static TypedColumn CreateTypedColumn(string name, string source, string type)
            {
                //            if (type == typeof(sbyte).ToString())
                //{
                //				return new Int8TypedColumn(name, source);
                //}
                //else if (type == typeof(short).ToString())
                //{
                //				return new Int16TypedColumn(name, source);
                //}
                //else if (type == typeof(int).ToString())
                //{
                //				return new Int32TypedColumn(name, source);
                //}
                //else if (type == typeof(long).ToString())
                //{
                //				return new Int64TypedColumn(name, source);
                //}
                //else if (type == typeof(byte).ToString())
                //{
                //				return new UInt8TypedColumn(name, source);
                //}
                //else if (type == typeof(ushort).ToString())
                //{
                //				return new UInt16TypedColumn(name, source);
                //}
                //else if (type == typeof(uint).ToString())
                //{
                //				return new UInt32TypedColumn(name, source);
                //}
                //else if (type == typeof(ulong).ToString())
                //{
                //				return new UInt64TypedColumn(name, source);
                //}
                //else if (type == typeof(float).ToString())
                //{
                //				return new FloatTypedColumn(name, source);
                //}
                //else if (type == typeof(double).ToString())
                //{
                //				return new DoubleTypedColumn(name, source);
                //}

                // TODO: Wrapper
                return null;

                throw new InvalidOperationException($"Column {name} has an unsupported type {type}.");
            }
        }

        internal abstract class TypedColumn<TSourceType, TOutputType> : TypedColumn
        {
            internal TypedColumn(string name, string source, string type) :
                base(name, source, type)
            {
            }

            internal abstract TOutputType Transform(TSourceType input);
            private protected abstract bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle);
            private protected abstract bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle);
            private protected abstract bool FitHelper(TransformerEstimatorSafeHandle estimator, TSourceType input, out FitResult fitResult, out IntPtr errorHandle);
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

                        // Fit the estimator
                        success = FitHelper(estimatorHandle, data.Current, out fitResult, out errorHandle);
                        if (!success)
                            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

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

                            data.Dispose();
                            data = input.GetColumn<TSourceType>(Source).GetEnumerator();
                            data.MoveNext();
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

                    return new TransformerEstimatorSafeHandle(transformer, DestroyTransformerHelper);
                }
            }
        }

        #endregion

        //#region Int8TypedColumn

        //internal sealed class Int8TypedColumn : TypedColumn<sbyte, double>
        //{
        //    internal Int8TypedColumn(string name, string source) :
        //        base(name, source, typeof(sbyte).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, sbyte input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(sbyte input)
        //    {
        //        sbyte interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, sbyte input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, sbyte input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        sbyte interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int8_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region Int16TypedColumn

        //internal sealed class Int16TypedColumn : TypedColumn<short, double>
        //{
        //    internal Int16TypedColumn(string name, string source) :
        //        base(name, source, typeof(short).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, short input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(short input)
        //    {
        //        short interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, short input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, short input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        short interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int16_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region Int32TypedColumn

        //internal sealed class Int32TypedColumn : TypedColumn<int, double>
        //{
        //    internal Int32TypedColumn(string name, string source) :
        //        base(name, source, typeof(int).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, int input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(int input)
        //    {
        //        int interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, int input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, int input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        int interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int32_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region Int64TypedColumn

        //internal sealed class Int64TypedColumn : TypedColumn<long, double>
        //{
        //    internal Int64TypedColumn(string name, string source) :
        //        base(name, source, typeof(long).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, long input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(long input)
        //    {
        //        long interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, long input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, long input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        long interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_int64_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region UInt8TypedColumn

        //internal sealed class UInt8TypedColumn : TypedColumn<byte, double>
        //{
        //    internal UInt8TypedColumn(string name, string source) :
        //        base(name, source, typeof(byte).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, byte input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(byte input)
        //    {
        //        byte interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, byte input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, byte input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        byte interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint8_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region UInt16TypedColumn

        //internal sealed class UInt16TypedColumn : TypedColumn<ushort, double>
        //{
        //    internal UInt16TypedColumn(string name, string source) :
        //        base(name, source, typeof(ushort).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ushort input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(ushort input)
        //    {
        //        ushort interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, ushort input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ushort input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        ushort interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint16_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region UInt32TypedColumn

        //internal sealed class UInt32TypedColumn : TypedColumn<uint, double>
        //{
        //    internal UInt32TypedColumn(string name, string source) :
        //        base(name, source, typeof(uint).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, uint input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(uint input)
        //    {
        //        uint interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, uint input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, uint input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        uint interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint32_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region UInt64TypedColumn

        //internal sealed class UInt64TypedColumn : TypedColumn<ulong, double>
        //{
        //    internal UInt64TypedColumn(string name, string source) :
        //        base(name, source, typeof(ulong).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, ulong input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(ulong input)
        //    {
        //        ulong interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, ulong input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, ulong input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        ulong interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_uint64_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region FloatTypedColumn

        //internal sealed class FloatTypedColumn : TypedColumn<float, double>
        //{
        //    internal FloatTypedColumn(string name, string source) :
        //        base(name, source, typeof(float).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, float input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(float input)
        //    {
        //        float interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, float input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, float input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        float interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_float_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

        //#region DoubleTypedColumn

        //internal sealed class DoubleTypedColumn : TypedColumn<double, double>
        //{
        //    internal DoubleTypedColumn(string name, string source) :
        //        base(name, source, typeof(double).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, double input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    double2: Delete transformed data
        //    internal unsafe override double Transform(double input)
        //    {
        //        double interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        double0: Invocation statements
        //        return output;

        //    }

        //    private protected override bool CreateEstimatorHelper(out IntPtr estimator, out IntPtr errorHandle)
        //    {
        //        /* Codegen: do any extra checks/paramters here */
        //        return CreateEstimatorNative(out estimator, out errorHandle);
        //    }

        //    private protected override bool CreateTransformerFromEstimatorHelper(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle) =>
        //        CreateTransformerFromEstimatorNative(estimator, out transformer, out errorHandle);

        //    private protected override bool DestroyEstimatorHelper(IntPtr estimator, out IntPtr errorHandle) =>
        //        DestroyEstimatorNative(estimator, out errorHandle);

        //    private protected override bool DestroyTransformerHelper(IntPtr transformer, out IntPtr errorHandle) =>
        //        DestroyTransformerNative(transformer, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        double interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "AnalyticalRollingWindowFeaturizer_double_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool OnDataCompletedNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool OnDataCompletedHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            OnDataCompletedNative(estimator, out errorHandle);

        //    public override void Dispose()
        //    {
        //        if (!TransformerHandler.IsClosed)
        //            TransformerHandler.Dispose();
        //    }

        //    public override Type ReturnType()
        //    {
        //        return typeof(double);
        //    }
        //}

        //#endregion

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
                _outputColumnName = _parent._options.TargetColumn + "_" + Enum.GetName(typeof(AnalyticalRollingWindowEstimator.AnalyticalRollingWindowCalculation), _parent._options.WindowCalculation);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                // To add future support for when this will do multiple columns at once, output will be a 2d vector so nothing will need to change when that is implemented.
                return new[] { new DataViewSchema.DetachedColumn(_outputColumnName, new VectorDataViewType(NumberDataViewType.Double, 1, (int)_parent._options.Horizon)) };
            }

            private Delegate MakeGetter<TSourceType, TOutputType>(DataViewRow input, int iinfo)
            {
                // TODO: wrapper
                //var inputColumn = input.Schema[_parent._columns[iinfo].Source];
                //var srcGetterScalar = input.GetGetter<TSourceType>(inputColumn);

                ValueGetter<TOutputType> result = (ref TOutputType dst) =>
                {
                    //TSourceType value = default;
                    //srcGetterScalar(ref value);

                    //dst = ((TypedColumn<TSourceType, TOutputType>)_parent._columns[iinfo]).Transform(value);
                    dst = default(TOutputType);
                };

                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                // TODO: wrapper
                //Type inputType = input.Schema[_parent._columns[iinfo].Source].Type.RawType;
                //Type outputType = _parent._columns[iinfo].ReturnType();

                Type inputType = typeof(int);
                Type outputType = typeof(VBuffer<double>);

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
