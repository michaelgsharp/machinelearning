// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TorchSharp.NasBert.Models;
using Microsoft.ML.TorchSharp.NasBert.Preprocessing;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using Microsoft.ML.TorchSharp.Utils;
using Microsoft.ML.TorchSharp.NasBert.Optimizers;
using Microsoft.ML;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.TorchSharp.Extensions;
using System.IO;
using System.CodeDom;
using System.Runtime.CompilerServices;

[assembly: LoadableClass(typeof(TorchSharpTransformer), null, typeof(SignatureLoadModel),
   TorchSharpTransformer.UserName, TorchSharpTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TorchSharpTransformer), null, typeof(SignatureLoadRowMapper),
   TorchSharpTransformer.UserName, TorchSharpTransformer.LoaderSignature)]

namespace Microsoft.ML.TorchSharp.NasBert
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a Deep Neural Network(DNN) to classify text.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [TextClassification](xref:Microsoft.ML.TorchSharpCatalog.TextClassification(Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers,Int32,System.String,System.String,System.String,System.String,Int32,Int32,Int32,Microsoft.ML.TorchSharp.NasBert.BertArchitecture,Microsoft.ML.IDataView)).
    ///
    /// ### Input and Output Columns
    /// The input label column data must be [key](xref:Microsoft.ML.Data.KeyDataViewType) type and the sentence columns must be of type<xref:Microsoft.ML.Data.TextDataViewType>.
    ///
    /// This trainer outputs the following columns:
    ///
    /// | Output Column Name | Column Type | Description|
    /// | -- | -- | -- |
    /// | `PredictedLabel` | [key](xref:Microsoft.ML.Data.KeyDataViewType) type | The predicted label's index. If its value is i, the actual label would be the i-th category in the key-valued input label type. |
    /// | `Score` | Vector of<xref:System.Single> | The scores of all classes.Higher value means higher probability to fall into the associated class. If the i-th element has the largest value, the predicted label index would be i.Note that i is zero-based index. |
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Multiclass classification |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.TorchSharp and libtorch-cpu or libtorch-cuda-11.3 or any of the OS specific variants. |
    /// | Exportable to ONNX | No |
    ///    ///
    /// ### Training Algorithm Details
    /// Trains a Deep Neural Network(DNN) by leveraging an existing pre-trained NAS-BERT roBERTa model for the purpose of classifying text.
    /// ]]>
    /// </format>
    /// </remarks>
    public abstract class TorchSharpTrainer : IEstimator<TorchSharpTransformer>
    {
        private readonly IHost _host;
        private readonly TorchSharpOptions _options;
        private TorchSharpTransformer _transformer;

        internal abstract class TorchSharpOptions : TransformInputBase
        {
            /// <summary>
            /// Number of samples to use for mini-batch training.
            /// </summary>
            public int BatchSize = 32;

            /// <summary>
            /// Stop training when reaching this number of epochs.
            /// </summary>
            public int MaxEpoch = 100;

            /// <summary>
            /// The validation set used while training to improve model quality.
            /// </summary>
            public IDataView ValidationSet = null;
        }

        internal TorchSharpTrainer(IHostEnvironment env, TorchSharpOptions options)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchSharpTrainer));
            Contracts.Assert(options.BatchSize > 0);
            Contracts.Assert(options.MaxEpoch > 0);
            _options = options;
        }

        public TorchSharpTransformer Fit(IDataView input)
        {
            using (var ch = _host.Start("TrainModel"))
            using (var pch = _host.StartProgressChannel("Training model"))
            {
                var header = new ProgressHeader(new[] { "Accuracy" }, null);
                var trainer = new Trainer(this, ch, input);
                pch.SetHeader(header, e => e.SetMetric(0, trainer.Accuracy));
                for (int i = 0; i < _options.MaxEpoch; i++)
                {
                    ch.Trace($"Starting epoch {i}");
                    trainer.Train(input);
                    ch.Trace($"Finished epoch {i}");
                    if (_options.ValidationSet != null)
                        trainer.Validate(pch, ch, i);
                }
                _transformer = new TorchSharpTransformer(_host, _options, trainer.Model, trainer.Tokenizer.Vocabulary);

                _transformer.GetOutputSchema(input.Schema);
            }
            return _transformer;
        }

        protected abstract class TrainerBase
        {
            protected readonly TorchSharpTrainer Parent;
            protected readonly IChannel Host;
            protected readonly torch.nn.Module Model;

            // Initialize getters during constructor
            // Initialize Model during custructor
            public TrainerBase(TorchSharpTrainer parent, IChannel ch)
            {
                Parent = parent;
                Host = ch;
            }

            protected abstract bool ValidateCore(DataViewRowCursor cursor, ref int numberCorrect, ref int numberOfRows);
            protected abstract bool TrainCore(DataViewRowCursor cursor);
            protected abstract DataViewRowCursor GetTrainingRowCursor(IDataView input);
            protected abstract DataViewRowCursor GetValidationRowCursor(IDataView input);
            protected abstract torch.nn.Module CreateModel();
            protected float Accuracy;

            public void Train(IDataView input)
            {
                // Set the torch random seed to match ML.NET if one was provided
                //if (((IHostEnvironmentInternal)_host).Seed.HasValue)
                torch.random.manual_seed(1);
                torch.cuda.manual_seed(1);

                // Get the cursor and the correct columns based on the inputs
                DataViewRowCursor cursor = GetTrainingRowCursor(input);

                var cursorValid = true;
                while (cursorValid)
                {
                    cursorValid = TrainCore(cursor);
                }
            }

            public void Validate(IProgressChannel pch, IChannel ch, int epoch)
            {
                Model.eval();

                var cursor = GetValidationRowCursor(Parent._options.ValidationSet);

                var numCorrect = 0;
                var numRows = 0;

                var cursorValid = true;
                while (cursorValid)
                {
                    cursorValid = ValidateCore(cursor, ref numCorrect, ref numRows);
                }
                Accuracy = numCorrect / (float)numRows;
                pch.Checkpoint(Accuracy);
                ch.Info($"Accuracy for epoch {epoch}: {Accuracy}");

                Model.train();
            }
        }

        private class Trainer
        {
            public BpeTokenizer Tokenizer;
            public BaseModule Model;
            public torch.Device Device;
            public BaseOptimizer Optimizer;
            public optim.lr_scheduler.LRScheduler LearningRateScheduler;
            private readonly TorchSharpTrainer _parent;
            public int Updates;
            public float Accuracy;

            public Trainer(TorchSharpTrainer parent, IChannel ch, IDataView input)
            {
            }

            public void Validate(IProgressChannel pch, IChannel ch, int epoch)
            {
                var validationSet = _parent._options.ValidationSet;
                Model.eval();

                DataViewRowCursor cursor = default;
                if (_parent._options.Sentence2ColumnName != default)
                    cursor = validationSet.GetRowCursor(validationSet.Schema[_parent._options.Sentence1ColumnName], validationSet.Schema[_parent._options.Sentence2ColumnName], validationSet.Schema[_parent._options.LabelColumnName]);
                else
                    cursor = validationSet.GetRowCursor(validationSet.Schema[_parent._options.Sentence1ColumnName], validationSet.Schema[_parent._options.LabelColumnName]);

                var sentence1Getter = cursor.GetGetter<ReadOnlyMemory<char>>(validationSet.Schema[_parent._options.Sentence1ColumnName]);
                var sentence2Getter = _parent._options.Sentence2ColumnName != default ? cursor.GetGetter<ReadOnlyMemory<char>>(validationSet.Schema[_parent._options.Sentence2ColumnName]) : default;
                var labelGetter = cursor.GetGetter<uint>(validationSet.Schema[_parent._options.LabelColumnName]);

                // Pre-allocate the memory so it's only done once (though this step needs to be optimized)
                List<Tensor> inputTensors = new List<Tensor>(_parent._options.BatchSize);
                List<long> targets = new List<long>(_parent._options.BatchSize);
                int numCorrect = 0;
                int numRows = 0;

                var cursorValid = true;
                while (cursorValid)
                {
                    cursorValid = ValidateStep(cursor, sentence1Getter, sentence2Getter, labelGetter, ref inputTensors, ref targets, ref numCorrect, ref numRows);
                }
                Accuracy = numCorrect / (float)numRows;
                pch.Checkpoint(Accuracy);
                ch.Info($"Accuracy for epoch {epoch}: {Accuracy}");

                Model.train();
            }

            private bool ValidateStep(DataViewRowCursor cursor,
                ValueGetter<ReadOnlyMemory<char>> sentence1Getter,
                ValueGetter<ReadOnlyMemory<char>> sentence2Getter,
                ValueGetter<uint> labelGetter,
                ref List<Tensor> inputTensors,
                ref List<long> targets,
                ref int numCorrect,
                ref int numRows)
            {
                // Make sure list is clear before use
                inputTensors.Clear();
                targets.Clear();
                using var disposeScope = torch.NewDisposeScope();
                var cursorValid = true;
                for (int i = 0; i < _parent._options.BatchSize && cursorValid; i++)
                {
                    cursorValid = cursor.MoveNext();
                    if (cursorValid)
                    {
                        inputTensors.Add(PrepareData(sentence1Getter, sentence2Getter));
                        uint target = default;
                        labelGetter(ref target);
                        // keys are 1 based but the model is 0 based
                        targets.Add(target - 1);
                    }
                    else
                    {
                        inputTensors.TrimExcess();
                        targets.TrimExcess();
                        if (inputTensors.Count() == 0)
                            return cursorValid;
                    }
                }

                using (torch.no_grad())
                {
                    var inputTensor = DataUtils.CollateTokens(inputTensors, Tokenizer.Vocabulary.PadIndex, device: Device);
                    var targetsTensor = tensor(targets, device: Device);
                    var logits = Model.forward(inputTensor);
                    var predictions = GetPredictions(logits);
                    var targetss = GetTargets(targetsTensor);
                    numCorrect = GetNumCorrect(predictions, targetss);
                    numRows = inputTensors.Count;
                }

                return cursorValid;
            }

            private static torch.Tensor GetPredictions(torch.Tensor logits)
            {
                logits = logits ?? throw new ArgumentNullException(nameof(logits));
                var (_, indexes) = logits.max(-1, false);
                return indexes;
            }

            private static torch.Tensor GetTargets(torch.Tensor labels)
            {
                return labels.view(-1);
            }

            private static int GetNumCorrect(torch.Tensor predictions, torch.Tensor targets)
            {
                predictions = predictions ?? throw new ArgumentNullException(nameof(predictions));
                return (int)predictions.eq(targets).sum().ToInt64();
            }

            private void OptimizeStep()
            {
                Optimizer.Step();
                LearningRateScheduler.step();
            }

            private torch.Tensor PrepareData(ValueGetter<ReadOnlyMemory<char>> sentence1Getter, ValueGetter<ReadOnlyMemory<char>> sentence2Getter)
            {
                ReadOnlyMemory<char> sentence1 = default;
                sentence1Getter(ref sentence1);
                Tensor t;
                if (sentence2Getter == default)
                {
                    t = torch.tensor((new[] { BpeTokenizer.InitToken }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString())).ToList(), device: Device);
                }
                else
                {

                    ReadOnlyMemory<char> sentence2 = default;
                    sentence2Getter(ref sentence2);

                    t = torch.tensor((new[] { BpeTokenizer.InitToken }).Concat(Tokenizer.EncodeToConverted(sentence1.ToString()))
                        .Concat(new[] { BpeTokenizer.SeperatorToken }).Concat(Tokenizer.EncodeToConverted(sentence2.ToString())).ToList(), device: Device);
                }

                if (t.NumberOfElements > 512)
                    t = t.slice(0, 0, 512, 1);

                return t;
            }
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            var metadata = new List<SchemaShape.Column>();
            metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                TextDataViewType.Instance, false));

            // Get label column for score column annotations. Already verified it exists.
            inputSchema.TryFindColumn(_options.LabelColumnName, out var labelCol);

            outColumns[_options.PredictionColumnName] = new SchemaShape.Column(_options.PredictionColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.UInt32, true, new SchemaShape(metadata.ToArray()));
            outColumns[_options.ScoreColumnName] = new SchemaShape.Column(_options.ScoreColumnName, SchemaShape.Column.VectorKind.Vector,
                NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol)));

            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(_options.Sentence1ColumnName, out var sentenceCol))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", _options.Sentence1ColumnName);
            if (sentenceCol.ItemType != TextDataViewType.Instance)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", _options.Sentence1ColumnName,
                    TextDataViewType.Instance.ToString(), sentenceCol.GetTypeString());

            if (!inputSchema.TryFindColumn(_options.LabelColumnName, out var labelCol))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "label", _options.LabelColumnName);
            if (labelCol.ItemType != NumberDataViewType.UInt32)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "label", _options.LabelColumnName,
                    NumberDataViewType.UInt32.ToString(), labelCol.GetTypeString());

            if (_options.Sentence2ColumnName != default)
            {
                if (!inputSchema.TryFindColumn(_options.Sentence2ColumnName, out var sentenceCol2))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", _options.Sentence2ColumnName);
                if (sentenceCol2.ItemType != TextDataViewType.Instance)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", _options.Sentence2ColumnName,
                        TextDataViewType.Instance.ToString(), sentenceCol2.GetTypeString());
            }
        }
    }

    public abstract class TorchSharpTransformer : RowToRowTransformerBase
    {
        internal const string LoadName = "NASBERTTrainer";
        internal const string UserName = "NASBERT Trainer";
        internal const string ShortName = "NASBERT";
        internal const string Summary = "NLP with NAS-BERT";

        private readonly Device _device;
        private readonly TextClassificationModel _model;
        private readonly Vocabulary _vocabulary;
        private readonly TorchSharpTrainer.Options _options;

        private readonly string _predictedLabelColumnName;
        private readonly string _scoreColumnName;

        public readonly SchemaShape.Column SentenceColumn;
        public readonly SchemaShape.Column SentenceColumn2;
        public readonly SchemaShape.Column LabelColumn;

        internal const string LoaderSignature = "NASBERT";

        internal TorchSharpTransformer(IHostEnvironment env, TorchSharpTrainer.Options options, TextClassificationModel model, Vocabulary vocabulary)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchSharpTransformer)))
        {
            _device = TorchUtils.InitializeDevice(env);

            _options = options;
            LabelColumn = new SchemaShape.Column(_options.LabelColumnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true);
            SentenceColumn = new SchemaShape.Column(_options.Sentence1ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            SentenceColumn2 = _options.Sentence2ColumnName == default ? default : new SchemaShape.Column(_options.Sentence2ColumnName, SchemaShape.Column.VectorKind.Scalar, TextDataViewType.Instance, false);
            _predictedLabelColumnName = _options.PredictionColumnName;
            _scoreColumnName = _options.ScoreColumnName;

            _vocabulary = vocabulary;
            _model = model;

            if (_device == CUDA)
                _model.cuda();
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureLoadModel.
        private static TorchSharpTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: id of label column name
            // int: id of score column name
            // int: id of output column name
            // int: id of sentence 1 column name
            // int: id of sentence 2 column name
            // int: number of classes
            var options = new TorchSharpTrainer.Options()
            {
                LabelColumnName = ctx.LoadString(),
                ScoreColumnName = ctx.LoadString(),
                PredictionColumnName = ctx.LoadString(),
                Sentence1ColumnName = ctx.LoadString(),
                Sentence2ColumnName = ctx.LoadStringOrNull(),
                NumberOfClasses = ctx.Reader.ReadInt32(),
            };

            var ch = env.Start("Load Model");
            var tokenizer = BpeTokenizer.GetInstance(ch);
            var vocabulary = tokenizer.Vocabulary;
            vocabulary.AddMaskSymbol();

            var model = new TextClassificationModel(options, vocabulary, options.NumberOfClasses);
            if (!ctx.TryLoadBinaryStream("TSModel", r => model.load(r)))
                throw env.ExceptDecode();

            return new TorchSharpTransformer(env, options, model, vocabulary);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);
            inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            var predLabelMetadata = new SchemaShape(labelCol.Annotations.Where(x => x.Name == AnnotationUtils.Kinds.KeyValues)
                .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            outColumns[_predictedLabelColumnName] = new SchemaShape.Column(_predictedLabelColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.UInt32, true, predLabelMetadata);

            outColumns[_scoreColumnName] = new SchemaShape.Column(_scoreColumnName, SchemaShape.Column.VectorKind.Vector,
                   NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol)));

            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(SentenceColumn.Name, out var sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", SentenceColumn.Name);
            if (!SentenceColumn.IsCompatibleWith(sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", SentenceColumn.Name,
                    SentenceColumn.GetTypeString(), sentenceCol.GetTypeString());

            if (_options.Sentence2ColumnName != default)
            {
                if (!inputSchema.TryFindColumn(SentenceColumn2.Name, out var sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", SentenceColumn2.Name);
                if (!SentenceColumn2.IsCompatibleWith(sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", SentenceColumn2.Name,
                        SentenceColumn2.GetTypeString(), sentenceCol2.GetTypeString());
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NAS-BERT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TorchSharpTransformer).Assembly.FullName);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: id of label column name
            // int: id of the score column name
            // int: id of output column name
            // int: id of sentence 1 column name
            // int: id of sentence 2 column name
            // int: number of classes

            ctx.SaveNonEmptyString(_options.LabelColumnName);
            ctx.SaveNonEmptyString(_options.ScoreColumnName);
            ctx.SaveNonEmptyString(_options.PredictionColumnName);
            ctx.SaveNonEmptyString(_options.Sentence1ColumnName);
            ctx.SaveStringOrNull(_options.Sentence2ColumnName);
            ctx.Writer.Write(_options.NumberOfClasses);

            ctx.SaveBinaryStream("TSModel", w =>
            {
                _model.save(w);
            });
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private readonly TorchSharpTransformer _parent;
            private readonly HashSet<int> _inputColIndices;
            private readonly DataViewSchema.Column _labelCol;
            private readonly DataViewSchema _inputSchema;
            private static readonly FuncInstanceMethodInfo1<Mapper, Delegate> _makeLabelAnnotationGetter
                = FuncInstanceMethodInfo1<Mapper, Delegate>.Create(target => target.GetLabelAnnotations<int>);

            public Mapper(TorchSharpTransformer parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _inputColIndices = new HashSet<int>();
                int col = 0;
                if (inputSchema.TryGetColumnIndex(parent._options.Sentence1ColumnName, out col))
                    _inputColIndices.Add(col);

                if (parent._options.Sentence2ColumnName != default)
                    if (inputSchema.TryGetColumnIndex(parent._options.Sentence2ColumnName, out col))
                        _inputColIndices.Add(col);

                _labelCol = inputSchema[_parent._options.LabelColumnName];
                _inputSchema = inputSchema;

                torch.random.manual_seed(1);
                torch.cuda.manual_seed(1);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[2];
                var keyType = _labelCol.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
                var getter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_makeLabelAnnotationGetter, this, keyType.ItemType.RawType);

                var meta = new DataViewSchema.Annotations.Builder();
                meta.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.ScoreColumnSetId, new KeyDataViewType(typeof(uint), _parent._options.NumberOfClasses), GetScoreColumnSetId(_inputSchema));
                meta.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.TrainingLabelValues, keyType, getter);
                meta.Add(AnnotationUtils.Kinds.SlotNames, keyType, getter);

                info[0] = new DataViewSchema.DetachedColumn(_parent._options.PredictionColumnName, new KeyDataViewType(typeof(uint), _parent._options.NumberOfClasses), _labelCol.Annotations);

                info[1] = new DataViewSchema.DetachedColumn(_parent._options.ScoreColumnName, new VectorDataViewType(NumberDataViewType.Single, _parent._options.NumberOfClasses), meta.ToAnnotations());
                return info;
            }

            private Delegate GetLabelAnnotations<T>()
            {
                return _labelCol.Annotations.GetGetter<VBuffer<T>>(_labelCol.Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
            }

            private ValueGetter<uint> GetScoreColumnSetId(DataViewSchema schema)
            {
                int c;
                var max = schema.GetMaxAnnotationKind(out c, AnnotationUtils.Kinds.ScoreColumnSetId);
                uint id = checked(max + 1);
                return
                    (ref uint dst) => dst = id;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
                => throw new NotImplementedException("This should never be called!");

            private Delegate CreateGetter(DataViewRow input, int iinfo, TensorCacher outputCacher)
            {
                var ch = Host.Start("Make Getter");
                if (iinfo == 0)
                    return MakePredictedLabelGetter(input, ch, outputCacher);
                else
                    return MakeScoreGetter(input, ch, outputCacher);
            }

            public override Delegate[] CreateGetters(DataViewRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Contracts.Assert(input.Schema == InputSchema);

                TensorCacher outputCacher = new TensorCacher();
                var ch = Host.Start("Make Getters");
                _parent._model.eval();

                int n = OutputColumns.Value.Length;
                var result = new Delegate[n];
                for (int i = 0; i < n; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    result[i] = CreateGetter(input, i, outputCacher);
                }
                disposer = () =>
                {
                    outputCacher.Dispose();
                };
                return result;
            }

            private Delegate MakeScoreGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<ReadOnlyMemory<char>> getSentence1 = default;
                ValueGetter<ReadOnlyMemory<char>> getSentence2 = default;

                BpeTokenizer tokenizer = BpeTokenizer.GetInstance(ch);

                getSentence1 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.SentenceColumn.Name]);
                if (_parent.SentenceColumn2.IsValid)
                    getSentence2 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.SentenceColumn2.Name]);

                ReadOnlyMemory<char> sentence1 = default;
                ReadOnlyMemory<char> sentence2 = default;

                ValueGetter<VBuffer<float>> score = (ref VBuffer<float> dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    var editor = VBufferEditor.Create(ref dst, _parent._options.NumberOfClasses);
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, ref tokenizer);
                    var values = outputCacher.Result.cpu().ToArray<float>();

                    for (var i = 0; i < values.Length; i++)
                    {
                        editor.Values[i] = values[i];
                    }
                    dst = editor.Commit();
                };

                return score;
            }

            private Delegate MakePredictedLabelGetter(DataViewRow input, IChannel ch, TensorCacher outputCacher)
            {
                ValueGetter<ReadOnlyMemory<char>> getSentence1 = default;
                ValueGetter<ReadOnlyMemory<char>> getSentence2 = default;

                BpeTokenizer tokenizer = BpeTokenizer.GetInstance(ch);

                getSentence1 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.SentenceColumn.Name]);
                if (_parent.SentenceColumn2.IsValid)
                    getSentence2 = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.SentenceColumn2.Name]);

                ReadOnlyMemory<char> sentence1 = default;
                ReadOnlyMemory<char> sentence2 = default;

                ValueGetter<UInt32> classification = (ref UInt32 dst) =>
                {
                    using var disposeScope = torch.NewDisposeScope();
                    UpdateCacheIfNeeded(input.Position, outputCacher, ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, ref tokenizer);
                    dst = (UInt32)outputCacher.Result.argmax(-1).cpu().item<long>() + 1;
                };

                return classification;
            }

            private IList<int> PrepInputTokens(ref ReadOnlyMemory<char> sentence1, ref ReadOnlyMemory<char> sentence2, ref ValueGetter<ReadOnlyMemory<char>> getSentence1, ref ValueGetter<ReadOnlyMemory<char>> getSentence2, ref BpeTokenizer tokenizer)
            {
                getSentence1(ref sentence1);
                if (getSentence2 == default)
                {
                    return new[] { BpeTokenizer.InitToken }.Concat(tokenizer.EncodeToConverted(sentence1.ToString())).ToList();
                }
                else
                {
                    getSentence2(ref sentence2);
                    return new[] { BpeTokenizer.InitToken }.Concat(tokenizer.EncodeToConverted(sentence1.ToString()))
                                              .Concat(new[] { BpeTokenizer.SeperatorToken }).Concat(tokenizer.EncodeToConverted(sentence2.ToString())).ToList();
                }
            }

            private Tensor PrepAndRunModel(IList<int> tokens)
            {
                using (torch.no_grad())
                {
                    var inputTensor = torch.tensor(tokens, device: _parent._device);
                    if (inputTensor.NumberOfElements > 512)
                        inputTensor = inputTensor.slice(0, 0, 512, 1);
                    inputTensor = inputTensor.reshape(1, inputTensor.NumberOfElements);
                    return _parent._model.forward(inputTensor);
                }
            }

            private class TensorCacher : IDisposable
            {
                public long Position;
                public Tensor Result;

                public TensorCacher()
                {
                    Position = -1;
                    Result = default;
                }

                private bool _isDisposed;

                public void Dispose()
                {
                    if (_isDisposed)
                        return;

                    Result?.Dispose();
                    _isDisposed = true;
                }
            }

            private void UpdateCacheIfNeeded(long position, TensorCacher outputCache, ref ReadOnlyMemory<char> sentence1, ref ReadOnlyMemory<char> sentence2, ref ValueGetter<ReadOnlyMemory<char>> getSentence1, ref ValueGetter<ReadOnlyMemory<char>> getSentence2, ref BpeTokenizer tokenizer)
            {
                if (outputCache.Position != position)
                {
                    outputCache.Result?.Dispose();
                    outputCache.Result = PrepAndRunModel(PrepInputTokens(ref sentence1, ref sentence2, ref getSentence1, ref getSentence2, ref tokenizer));
                    outputCache.Result.MoveToOuterDisposeScope();
                    outputCache.Position = position;
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                return col => (activeOutput(0) || activeOutput(1)) && _inputColIndices.Any(i => i == col);
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }
}
