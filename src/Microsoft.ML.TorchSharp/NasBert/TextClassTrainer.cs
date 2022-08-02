// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TorchSharp.NasBert.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.TorchSharp.NasBert
{
    public class TextClassTrainer : NasBertBaseTrainer
    {

        internal class TextClassTrainerOptions : NasBertBaseTrainerOptions
        {
            /// <summary>
            /// The Prediction column name.
            /// </summary>
            public string PredictionColumnName = DefaultColumnNames.PredictedLabel;
        }

        internal TextClassTrainer(IHostEnvironment env, TextClassTrainerOptions options) : base(env, options)
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            var metadata = new List<SchemaShape.Column>();
            metadata.Add(new SchemaShape.Column(AnnotationUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                TextDataViewType.Instance, false));

            // Get label column for score column annotations. Already verified it exists.
            inputSchema.TryFindColumn(Options.LabelColumnName, out var labelCol);

            outColumns[(Options as TextClassTrainerOptions).PredictionColumnName] = new SchemaShape.Column((Options as TextClassTrainerOptions).PredictionColumnName, SchemaShape.Column.VectorKind.Scalar,
                    NumberDataViewType.UInt32, true, new SchemaShape(metadata.ToArray()));
            outColumns[Options.ScoreColumnName] = new SchemaShape.Column(Options.ScoreColumnName, SchemaShape.Column.VectorKind.Vector,
                NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol)));

            return new SchemaShape(outColumns.Values);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(Options.Sentence1ColumnName, out var sentenceCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", Options.Sentence1ColumnName);
            if (sentenceCol.ItemType != TextDataViewType.Instance)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence", Options.Sentence1ColumnName,
                    TextDataViewType.Instance.ToString(), sentenceCol.GetTypeString());

            if (!inputSchema.TryFindColumn(Options.LabelColumnName, out var labelCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Options.LabelColumnName);
            if (labelCol.ItemType != NumberDataViewType.UInt32)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", Options.LabelColumnName,
                    NumberDataViewType.UInt32.ToString(), labelCol.GetTypeString());

            if (Options.Sentence2ColumnName != default)
            {
                if (!inputSchema.TryFindColumn(Options.Sentence2ColumnName, out var sentenceCol2))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", Options.Sentence2ColumnName);
                if (sentenceCol2.ItemType != TextDataViewType.Instance)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "sentence2", Options.Sentence2ColumnName,
                        TextDataViewType.Instance.ToString(), sentenceCol2.GetTypeString());
            }
        }

        protected override TrainerBase GetTrainer(IChannel ch, IDataView dataView)
        {
            return new TextTrainer(this, ch, dataView);
        }

        protected override TorchSharpTransformer CreateTransformer(TrainerBase trainer)
        {
            throw new NotImplementedException();
        }

        private sealed class TextTrainer : Trainer
        {
            private readonly TextClassTrainer _parent;
            private ValueGetter<ReadOnlyMemory<char>> _sentence1Getter;
            private ValueGetter<ReadOnlyMemory<char>> _sentence2Getter;
            private ValueGetter<uint> _labelGetter;

            public TextTrainer(TextClassTrainer parent, IChannel ch, IDataView dataView) : base(parent, ch, dataView)
            {
                _parent = parent;
            }

            protected override int GetRowCountAndSetLabelCount(IDataView input)
            {
                var labelCol = input.GetColumn<uint>(_parent.Options.LabelColumnName);
                var rowCount = 0;
                var uniqueLabels = new HashSet<uint>();

                foreach (var label in labelCol)
                {
                    rowCount++;
                    uniqueLabels.Add(label);
                }

                _parent.Options.NumberOfClasses = uniqueLabels.Count;
                return rowCount;
            }

            protected override DataViewRowCursor GetTrainingRowCursor(IDataView input)
            {
                InputTensors = new List<Tensor>(_parent.Options.BatchSize);

                DataViewRowCursor cursor = default;

                if (_parent.Options.Sentence2ColumnName != default)
                    cursor = input.GetRowCursor(input.Schema[_parent.Options.Sentence1ColumnName], input.Schema[_parent.Options.Sentence2ColumnName], input.Schema[_parent.Options.LabelColumnName]);
                else
                    cursor = input.GetRowCursor(input.Schema[_parent.Options.Sentence1ColumnName], input.Schema[_parent.Options.LabelColumnName]);

                _sentence1Getter = cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.Options.Sentence1ColumnName]);
                _sentence2Getter = _parent.Options.Sentence2ColumnName != default ? cursor.GetGetter<ReadOnlyMemory<char>>(input.Schema[_parent.Options.Sentence2ColumnName]) : default;
                _labelGetter = cursor.GetGetter<UInt32>(input.Schema[_parent.Options.LabelColumnName]);

                return cursor;
            }

            protected override DataViewRowCursor GetValidationRowCursor(IDataView input)
            {
                return GetTrainingRowCursor(input);
            }
        }
    }

    public class TextClassTransformer : NasBertTransformerBase
    {
        internal TextClassTransformer(IHostEnvironment env, TextClassTrainer.TextClassTrainerOptions options, nn.Module model) : base(env, options, model)
        {
        }

        private protected override IRowMapper GetRowMapper(TorchSharpTransformer parent, DataViewSchema schema)
        {
            throw new NotImplementedException();
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
        }

        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureLoadModel.
        private static TextClassificationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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
            var options = new TextClassificationTrainer.Options()
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

            BinarySaver saver = new BinarySaver(env, new BinarySaver.Arguments());
            DataViewType type;
            object value;
            env.CheckDecode(saver.TryLoadTypeAndValue(ctx.Reader.BaseStream, out type, out value));
            var vecType = type as VectorDataViewType;
            env.CheckDecode(vecType != null);
            env.CheckDecode(value != null);
            var labelGetter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_decodeInitMethodInfo, vecType.ItemType.RawType, value);

            var meta = new DataViewSchema.Annotations.Builder();
            meta.Add(AnnotationUtils.Kinds.KeyValues, type, labelGetter);

            var labelCol = new DataViewSchema.DetachedColumn(options.LabelColumnName, type, meta.ToAnnotations());

            return new TextClassificationTransformer(env, options, model, labelCol);
        }

        private new sealed class Mapper : NasBertTransformerBase.Mapper
        {
            private readonly TextClassTransformer _parent;
            private readonly HashSet<int> _inputColIndices;
            private readonly DataViewSchema _inputSchema;

            private static readonly FuncInstanceMethodInfo1<Mapper, DataViewSchema.DetachedColumn, Delegate> _makeLabelAnnotationGetter
                = FuncInstanceMethodInfo1<Mapper, DataViewSchema.DetachedColumn, Delegate>.Create(target => target.GetLabelAnnotations<int>);


            public Mapper(TextClassTransformer parent, DataViewSchema inputSchema) :
                base(parent, inputSchema)
            {
                _parent = parent;
                _inputColIndices = new HashSet<int>();
                if (inputSchema.TryGetColumnIndex(parent.Options.Sentence1ColumnName, out var col))
                    _inputColIndices.Add(col);

                if (parent._options.Sentence2ColumnName != default)
                    if (inputSchema.TryGetColumnIndex(parent._options.Sentence2ColumnName, out col))
                        _inputColIndices.Add(col);

                _inputSchema = inputSchema;

                torch.random.manual_seed(1);
                torch.cuda.manual_seed(1);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var info = new DataViewSchema.DetachedColumn[2];
                var keyType = _parent.LabelColumn.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type as VectorDataViewType;
                var getter = Microsoft.ML.Internal.Utilities.Utils.MarshalInvoke(_makeLabelAnnotationGetter, this, keyType.ItemType.RawType, _parent.LabelColumn);


                var meta = new DataViewSchema.Annotations.Builder();
                meta.Add(AnnotationUtils.Kinds.ScoreColumnKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreColumnKind.MulticlassClassification.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.ScoreColumnSetId, new KeyDataViewType(typeof(uint), _parent._options.NumberOfClasses), GetScoreColumnSetId(_inputSchema));
                meta.Add(AnnotationUtils.Kinds.ScoreValueKind, TextDataViewType.Instance, (ref ReadOnlyMemory<char> value) => { value = AnnotationUtils.Const.ScoreValueKind.Score.AsMemory(); });
                meta.Add(AnnotationUtils.Kinds.TrainingLabelValues, keyType, getter);
                meta.Add(AnnotationUtils.Kinds.SlotNames, keyType, getter);

                var labelBuilder = new DataViewSchema.Annotations.Builder();
                labelBuilder.Add(AnnotationUtils.Kinds.KeyValues, keyType, getter);

                info[0] = new DataViewSchema.DetachedColumn(_parent._options.PredictionColumnName, new KeyDataViewType(typeof(uint), _parent._options.NumberOfClasses), labelBuilder.ToAnnotations());

                info[1] = new DataViewSchema.DetachedColumn(_parent._options.ScoreColumnName, new VectorDataViewType(NumberDataViewType.Single, _parent._options.NumberOfClasses), meta.ToAnnotations());
                return info;
            }

            private Delegate GetLabelAnnotations<T>(DataViewSchema.DetachedColumn labelCol)
            {
                return labelCol.Annotations.GetGetter<VBuffer<T>>(labelCol.Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
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
                    var inputTensor = torch.tensor(tokens, device: _parent.Device);
                    if (inputTensor.NumberOfElements > 512)
                        inputTensor = inputTensor.slice(0, 0, 512, 1);
                    inputTensor = inputTensor.reshape(1, inputTensor.NumberOfElements);
                    return _parent.Model.forward(inputTensor);
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
