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

[assembly: LoadableClass(typeof(LagLeadOperatorTransformer), null, typeof(SignatureLoadModel),
    LagLeadOperatorTransformer.UserName, LagLeadOperatorTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(LagLeadOperatorTransformer), null, typeof(SignatureLoadRowMapper),
LagLeadOperatorTransformer.UserName, LagLeadOperatorTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(LagLeadOperatorEntrypoint))]

namespace Microsoft.ML.Featurizers
{
    public static class LagLeadOperatorExtensionClass
    {
        public static LagLeadOperatorEstimator LagLeadOperatorTransformer(this TransformsCatalog catalog, string[] grainColumns, string targetColumn, UInt32 horizon, long[] offsets)
        {
            var options = new LagLeadOperatorEstimator.Options {
                GrainColumns = grainColumns,
                TargetColumn = targetColumn,
                Horizon = horizon,
                Offsets = offsets
            };

            return new LagLeadOperatorEstimator(CatalogUtils.GetEnvironment(catalog), options);
        }
    }

    public class LagLeadOperatorEstimator : IEstimator<LagLeadOperatorTransformer>
    {
        private Options _options;
        private readonly IHost _host;

        /* Codegen: Add additional needed class members here */

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

            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "Lag and Lead offset to use. A negative number is a lag, positive is a lead",
                Name = "offsets", ShortName = "off", SortOrder = 3)]
            public long[] Offsets;
        }

        #endregion

        internal LagLeadOperatorEstimator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(LagLeadOperatorEstimator));
            Contracts.CheckNonEmpty(options.GrainColumns, nameof(options.GrainColumns));

            _options = options;
        }

        public LagLeadOperatorTransformer Fit(IDataView input)
        {
            return new LagLeadOperatorTransformer(_host, input, _options);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.ToDictionary(x => x.Name);

            var inputColumn = columns[_options.TargetColumn];

            if (!LagLeadOperatorTransformer.TypedColumn.IsColumnTypeSupported(inputColumn.ItemType.RawType))
                throw new InvalidOperationException($"Type {inputColumn.ItemType.RawType.ToString()} for column {_options.TargetColumn} not a supported type.");

            columns[_options.TargetColumn] = new SchemaShape.Column(_options.TargetColumn + "Laglead", VectorKind.Vector,
                NumberDataViewType.Double, false, inputColumn.Annotations);

            return new SchemaShape(columns.Values);
        }
    }

    public sealed class LagLeadOperatorTransformer : RowToRowTransformerBase, IDisposable
    {
        #region Class data members

        internal const string Summary = "uses the offset list to create lags and leads";
        internal const string UserName = "LagLeadOperatorTransformer";
        internal const string ShortName = "LagLeadOperatorTransformer";
        internal const string LoadName = "LagLeadOperatorTransformer";
        internal const string LoaderSignature = "LagLeadOperatorTransformer";

        private TypedColumn _column;
        private LagLeadOperatorEstimator.Options _options;

        #endregion

        internal LagLeadOperatorTransformer(IHostEnvironment host, IDataView input, LagLeadOperatorEstimator.Options options) :
            base(host.Register(nameof(LagLeadOperatorTransformer)))
        {
            var schema = input.Schema;
            _options = options;

            _column = TypedColumn.CreateTypedColumn(_options.TargetColumn + "Laglead", _options.TargetColumn, schema[_options.TargetColumn].Type.RawType.ToString());
            // TODO: wrapper
            //_column.CreateTransformerFromEstimator(input);
        }

        // Factory method for SignatureLoadModel.
        internal LagLeadOperatorTransformer(IHostEnvironment host, ModelLoadContext ctx) :
            base(host.Register(nameof(LagLeadOperatorTransformer)))
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

            _options = new LagLeadOperatorEstimator.Options();

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
            => new LagLeadOperatorTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private static VersionInfo GetVersionInfo()
        {
            /* Codegen: Change these as needed */
            return new VersionInfo(
                modelSignature: "LAGLED T",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LagLeadOperatorTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            /* Codegen: Edit this format as needed */
            // *** Binary format ***
            // int number of column pairs
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
    //            if (type == typeof(double).ToString()) {
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

        //#region DoubleTypedColumn

        //internal sealed class DoubleTypedColumn : TypedColumn<double, TODO1>
        //{
        //    internal DoubleTypedColumn(string name, string source) :
        //        base(name, source, typeof(double).ToString())
        //    {
        //    }

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CreateEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateEstimatorNative(/* Codegen: Add additional parameters here */ out IntPtr estimator, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_DestroyEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyEstimatorNative(IntPtr estimator, out IntPtr errorHandle); // Should ONLY be called by safe handle

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CreateTransformerFromEstimator", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerFromEstimatorNative(TransformerEstimatorSafeHandle estimator, out IntPtr transformer, out IntPtr errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_DestroyTransformer", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool DestroyTransformerNative(IntPtr transformer, out IntPtr errorHandle);
        //    internal override void CreateTransformerFromEstimator(IDataView input)
        //    {
        //        TransformerHandler = CreateTransformerFromEstimatorBase(input);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CreateTransformerFromSavedData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool CreateTransformerFromSavedDataNative(byte* rawData, IntPtr bufferSize, out IntPtr transformer, out IntPtr errorHandle);
        //    private protected override unsafe void CreateTransformerFromSavedDataHelper(byte* rawData, IntPtr dataSize)
        //    {
        //        var result = CreateTransformerFromSavedDataNative(rawData, dataSize, out IntPtr transformer, out IntPtr errorHandle);
        //        if (!result)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TransformerHandler = new TransformerEstimatorSafeHandle(transformer, DestroyTransformerNative);
        //    }

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_Transform", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool TransformDataNative(TransformerEstimatorSafeHandle transformer, double input, TODO8: Parameter decl, out IntPtr errorHandle);
        //    TODO12: Delete transformed data
        //    internal unsafe override TODO1 Transform(double input)
        //    {
        //        double interopInput = input;
        //        var success = TransformDataNative(TransformerHandler, interopInput, TODO8: Parameter decl, out IntPtr errorHandle);
        //        if (!success)
        //            throw new Exception(GetErrorDetailsAndFreeNativeMemory(errorHandle));

        //        TODO10: Invocation statements
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

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_Fit", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static unsafe extern bool FitNative(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle);
        //    private protected unsafe override bool FitHelper(TransformerEstimatorSafeHandle estimator, double input, out FitResult fitResult, out IntPtr errorHandle)
        //    {
        //        double interopInput = input;
        //        return FitNative(estimator, interopInput, out fitResult, out errorHandle);

        //    }

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CompleteTraining", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CompleteTrainingNative(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle);
        //    private protected override bool CompleteTrainingHelper(TransformerEstimatorSafeHandle estimator, out IntPtr errorHandle) =>
        //            CompleteTrainingNative(estimator, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_CreateTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool CreateTransformerSaveDataNative(TransformerEstimatorSafeHandle transformer, out IntPtr buffer, out IntPtr bufferSize, out IntPtr error);
        //    private protected override bool CreateTransformerSaveDataHelper(out IntPtr buffer, out IntPtr bufferSize, out IntPtr errorHandle) =>
        //        CreateTransformerSaveDataNative(TransformerHandler, out buffer, out bufferSize, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_GetState", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        //    private static extern bool GetStateNative(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle);
        //    private protected override bool GetStateHelper(TransformerEstimatorSafeHandle estimator, out TrainingState trainingState, out IntPtr errorHandle) =>
        //        GetStateNative(estimator, out trainingState, out errorHandle);

        //    [DllImport("Featurizers", EntryPoint = "LagLeadOperatorFeaturizer_double_OnDataCompleted", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
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
        //        return typeof(TODO1);
        //    }
        //}

        //#endregion

        #endregion

        private sealed class Mapper : MapperBase
        {
            #region Class members

            private readonly LagLeadOperatorTransformer _parent;
            /* Codegen: add any extra class members here */

            #endregion

            public Mapper(LagLeadOperatorTransformer parent, DataViewSchema inputSchema) :
                base(parent.Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                // TODO: wrapper
                return null;
                //return _parent._columns.Select(x => new DataViewSchema.DetachedColumn(x.Name, ColumnTypeExtensions.PrimitiveTypeFromType(x.ReturnType()))).ToArray();
            }

            private Delegate MakeGetter<TSourceType, TOutputType>(DataViewRow input, int iinfo)
            {
                // TODO: wrapper
                //var inputColumn = input.Schema[_parent._columns[iinfo].Source];
                //var srcGetterScalar = input.GetGetter<TSourceType>(inputColumn);

                ValueGetter<TOutputType> result = (ref TOutputType dst) => {
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

    internal static class LagLeadOperatorEntrypoint
    {
        [TlcModule.EntryPoint(Name = "Transforms.LagLeadOperator",
            Desc = LagLeadOperatorTransformer.Summary,
            UserName = LagLeadOperatorTransformer.UserName,
            ShortName = LagLeadOperatorTransformer.ShortName)]
        public static CommonOutputs.TransformOutput LagLeadOperator(IHostEnvironment env, LagLeadOperatorEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, LagLeadOperatorTransformer.ShortName, input);
            var xf = new LagLeadOperatorEstimator(h, input).Fit(input.Data).Transform(input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }
    }
}
