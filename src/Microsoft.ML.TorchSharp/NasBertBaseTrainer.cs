// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TorchSharp.NasBert;
using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using System.Linq;
using Microsoft.ML.TorchSharp.NasBert.Preprocessing;
using Microsoft.ML.TorchSharp.NasBert.Models;
using System.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.TorchSharp.Utils;
using Microsoft.ML.TorchSharp.NasBert.Optimizers;

namespace Microsoft.ML.TorchSharp
{
    public abstract class NasBertBaseTrainer : TorchSharpTrainer
    {
        internal readonly NasBertBaseTrainerOptions Options;
        private const string ModelUrl = "models/NasBert2000000.tsm";

        internal class NasBertBaseTrainerOptions : TorchSharpOptions
        {
            /// <summary>
            /// The label column name.
            /// </summary>
            public string LabelColumnName = DefaultColumnNames.Label;

            /// <summary>
            /// The Score column name.
            /// </summary>
            public string ScoreColumnName = DefaultColumnNames.Score;

            /// <summary>
            /// The first sentence column.
            /// </summary>
            public string Sentence1ColumnName = "Sentence";

            /// <summary>
            /// The second sentence column.
            /// </summary>
            public string Sentence2ColumnName = default;

            /// <summary>
            /// Whether to freeze encoder parameters.
            /// </summary>
            public bool FreezeEncoder = false;

            /// <summary>
            /// Whether to freeze transfer module parameters.
            /// </summary>
            public bool FreezeTransfer = false;

            /// <summary>
            /// Whether to train layer norm parameters.
            /// </summary>
            public bool LayerNormTraining = false;

            /// <summary>
            /// Whether to apply layer normalization before each encoder block.
            /// </summary>
            public bool EncoderNormalizeBefore = true;

            /// <summary>
            /// Dropout rate for general situations. Should be within [0, 1).
            /// </summary>
            public double Dropout = .1;

            /// <summary>
            /// Dropout rate for attention weights. Should be within [0, 1).
            /// </summary>
            public double AttentionDropout = .1;

            /// <summary>
            /// Dropout rate after activation functions in FFN layers. Should be within [0, 1).
            /// </summary>
            public double ActivationDropout = 0;

            /// <summary>
            /// Whether to use dynamic dropout.
            /// </summary>
            public bool DynamicDropout = false;

            /// <summary>
            /// Dropout rate in the masked language model pooler layers. Should be within [0, 1).
            /// </summary>
            public double PoolerDropout = 0;

            /// <summary>
            /// The start learning rate for polynomial decay scheduler.
            /// </summary>
            public double StartLearningRateRatio = .1;

            /// <summary>
            /// The final learning rate for polynomial decay scheduler.
            /// </summary>
            public double FinalLearningRateRatio = .1;

            /// <summary>
            /// Betas for Adam optimizer.
            /// </summary>
            public IReadOnlyList<double> AdamBetas = new List<double> { .9, .999 };

            /// <summary>
            /// Epsilon for Adam optimizer.
            /// </summary>
            public double AdamEps = 1e-8;

            /// <summary>
            /// Coefficiency of weight decay. Should be within [0, +Inf).
            /// </summary>
            public double WeightDecay = 0;

            /// <summary>
            /// The clipping threshold of gradients. Should be within [0, +Inf). 0 means not to clip norm.
            /// </summary>
            public double ClipNorm = 25;

            /// <summary>
            /// Proportion of warmup steps for polynomial decay scheduler.
            /// </summary>
            public double WarmupRatio = .06;

            /// <summary>
            /// Number of classes for the data.
            /// </summary>
            internal int NumberOfClasses;

            /// <summary>
            /// Learning rate for the first N epochs; all epochs >N using LR_N.
            /// Note: this may be interpreted differently depending on the scheduler.
            /// </summary>
            internal List<double> LearningRate = new List<double> { 1e-4 };

            /// <summary>
            /// The index numbers of model architecture. Fixed by the TorchSharp model.
            /// </summary>
            internal IReadOnlyList<int> Arches = new int[] { 9, 11, 7, 0, 0, 0, 11, 11, 7, 0, 0, 0, 9, 7, 11, 0, 0, 0, 10, 7, 9, 0, 0, 0 };

            /// <summary>
            /// Task type, which is related to the model head.
            /// </summary>
            internal BertTaskType TaskType = BertTaskType.TextClassification;

            /// <summary>
            /// Maximum length of a sample. Set by the TorchSharp model.
            /// </summary>
            internal int MaxSequenceLength = 512;

            /// <summary>
            /// Number of the embedding dimensions. Should be positive. Set by the TorchSharp model.
            /// </summary>
            internal int EmbeddingDim = 64;

            /// <summary>
            /// Number of encoder layers. Set by the TorchSharp model.
            /// </summary>
            internal int EncoderLayers = 24;

            /// <summary>
            ///  Number of the output dimensions of encoder. Should be positive. Set by the TorchSharp model. 3 * EmbeddingDim
            /// </summary>
            internal int EncoderOutputDim = 192;

            /// <summary>
            /// The activation function to use for general situations. Set by the TorchSharp model.
            /// </summary>
            internal string ActivationFunction = "gelu";

            /// <summary>
            /// The activation function to use for pooler layers. Set by the TorchSharp model.
            /// </summary>
            internal string PoolerActivationFunction = "tanh";

            /// <summary>
            /// Reduction of criterion loss function. Set by the TorchSharp model.
            /// </summary>
            internal torch.nn.Reduction Reduction = Reduction.Sum;
        }

        internal NasBertBaseTrainer(IHostEnvironment env, NasBertBaseTrainerOptions options) : base(env, options)
        {
            Contracts.AssertValue(options.Sentence1ColumnName);
            Contracts.AssertValue(options.LabelColumnName);
            Contracts.AssertValue(options.ScoreColumnName);

            Options = options;
        }

        protected abstract class Trainer : TrainerBase
        {
            private protected BpeTokenizer Tokenizer;
            private readonly NasBertBaseTrainer _parent;
            internal BaseOptimizer Optimizer;
            public optim.lr_scheduler.LRScheduler LearningRateScheduler;

            public Trainer(NasBertBaseTrainer parent, IChannel ch, IDataView dataView) : base(parent, ch, dataView)
            {
                _parent = parent;

                Tokenizer = BpeTokenizer.GetInstance(ch);

                // Initialize the vocab
                Tokenizer = BpeTokenizer.GetInstance(ch);

                // Get row count and figure out num of unique labels
                var rowCount = GetRowCountAndSetLabelCount(dataView);

                // Figure out if we are running on GPU or CPU
                Device = TorchUtils.InitializeDevice(_parent.Host);

                // Move to GPU if we are running there
                if (Device == CUDA)
                    Model.cuda();

                // Get the paramters that need optimization and set up the optimizer
                var parameters = Model.parameters().Where(p => p.requires_grad);
                Optimizer = BaseOptimizer.GetOptimizer(_parent.Options, parameters);
                LearningRateScheduler = torch.optim.lr_scheduler.OneCycleLR(
                   Optimizer.Optimizer,
                   max_lr: _parent.Options.LearningRate[0],
                   total_steps: ((rowCount / _parent.Options.BatchSize) + 1) * _parent.Options.MaxEpoch,
                   pct_start: _parent.Options.WarmupRatio,
                   anneal_strategy: torch.optim.lr_scheduler.impl.OneCycleLR.AnnealStrategy.Linear,
                   div_factor: 1.0 / _parent.Options.StartLearningRateRatio,
                   final_div_factor: 1.0 / _parent.Options.FinalLearningRateRatio);
            }

            protected abstract int GetRowCountAndSetLabelCount(IDataView input);

            protected override Module CreateModel()
            {
                var vocabulary = Tokenizer.Vocabulary;
                vocabulary.AddMaskSymbol();

                var model = new TextClassificationModel(_parent.Options, vocabulary, _parent.Options.NumberOfClasses);
                model.GetEncoder().load(GetModelPath());

                return model;
            }

            private string GetModelPath()
            {
                var destDir = Path.Combine(((IHostEnvironmentInternal)_parent.Host).TempFilePath, "mlnet");
                var destFileName = ModelUrl.Split('/').Last();

                Directory.CreateDirectory(destDir);

                string relativeFilePath = Path.Combine(destDir, destFileName);

                int timeout = 10 * 60 * 1000;
                using (var ch = (_parent.Host as IHostEnvironment).Start("Ensuring model file is present."))
                {
                    var ensureModel = ResourceManagerUtils.Instance.EnsureResourceAsync(_parent.Host, ch, ModelUrl, destFileName, destDir, timeout);
                    ensureModel.Wait();
                    var errorResult = ResourceManagerUtils.GetErrorMessage(out var errorMessage, ensureModel.Result);
                    if (errorResult != null)
                    {
                        var directory = Path.GetDirectoryName(errorResult.FileName);
                        var name = Path.GetFileName(errorResult.FileName);
                        throw ch.Except($"{errorMessage}\nmodel file could not be downloaded!");
                    }
                }

                return relativeFilePath;
            }

            protected override bool TrainCore(DataViewRowCursor cursor)
            {
                throw new NotImplementedException();
            }

            protected override bool ValidateCore(DataViewRowCursor cursor, ref int numberCorrect, ref int numberOfRows)
            {
                throw new NotImplementedException();
            }
        }
    }

    public abstract class NasBertTransformerBase : TorchSharpTransformer
    {
        internal NasBertTransformerBase(IHostEnvironment env, NasBertBaseTrainer.NasBertBaseTrainerOptions options, Module model) : base(env, options, model)
        {
        }

        private new abstract class Mapper : TorchSharpTransformer.Mapper
        {
            protected Mapper(NasBertTransformerBase parent, DataViewSchema inputSchema) : base(parent, inputSchema)
            {
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                throw new NotImplementedException();
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                throw new NotImplementedException();
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                throw new NotImplementedException();
            }
        }

    }
}
