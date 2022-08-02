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
        protected readonly IHost Host;
        private readonly TorchSharpOptions _options;

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
            Host = Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchSharpTrainer));
            Contracts.Assert(options.BatchSize > 0);
            Contracts.Assert(options.MaxEpoch > 0);
            _options = options;
        }

        public TorchSharpTransformer Fit(IDataView input)
        {
            TorchSharpTransformer transformer = default;

            using (var ch = Host.Start("TrainModel"))
            using (var pch = Host.StartProgressChannel("Training model"))
            {
                var header = new ProgressHeader(new[] { "Accuracy" }, null);
                var trainer = GetTrainer(ch, input);
                pch.SetHeader(header, e => e.SetMetric(0, trainer.Accuracy));
                for (int i = 0; i < _options.MaxEpoch; i++)
                {
                    ch.Trace($"Starting epoch {i}");
                    trainer.Train(input);
                    ch.Trace($"Finished epoch {i}");
                    if (_options.ValidationSet != null)
                        trainer.Validate(pch, ch, i);
                }
                transformer = CreateTransformer(trainer);

                transformer.GetOutputSchema(input.Schema);
            }
            return transformer;
        }

        protected abstract TorchSharpTransformer CreateTransformer(TrainerBase trainer);

        protected abstract TrainerBase GetTrainer(IChannel ch, IDataView dataView);

        public abstract SchemaShape GetOutputSchema(SchemaShape inputSchema);

        protected abstract class TrainerBase
        {
            protected readonly TorchSharpTrainer Parent;
            protected readonly IChannel Host;
            protected readonly IDataView TrainingDataView;
            public torch.Device Device;

            public readonly torch.nn.Module Model;
            public float Accuracy;
            protected List<Tensor> InputTensors;

            // Initialize getters during constructor
            // Initialize Model during custructor
            public TrainerBase(TorchSharpTrainer parent, IChannel ch, IDataView dataView)
            {
                Parent = parent;
                Host = ch;
                TrainingDataView = dataView;
                Model = CreateModel();
            }

            protected abstract bool ValidateCore(DataViewRowCursor cursor, ref int numberCorrect, ref int numberOfRows);
            protected abstract bool TrainCore(DataViewRowCursor cursor);
            protected abstract DataViewRowCursor GetTrainingRowCursor(IDataView input);
            protected abstract DataViewRowCursor GetValidationRowCursor(IDataView input);
            protected abstract torch.nn.Module CreateModel();

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
    }

    public abstract class TorchSharpTransformer : RowToRowTransformerBase
    {
        internal readonly Device Device;
        internal readonly torch.nn.Module Model;
        internal readonly TorchSharpTrainer.TorchSharpOptions Options;

        internal TorchSharpTransformer(IHostEnvironment env, TorchSharpTrainer.TorchSharpOptions options, torch.nn.Module model)
           : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchSharpTransformer)))
        {
            Device = TorchUtils.InitializeDevice(env);

            Options = options;
            Model = model;

            if (Device == CUDA)
                Model.cuda();
        }

        private protected void LoadModel(ModelLoadContext ctx, IHostEnvironment env, torch.nn.Module model)
        {
            if (!ctx.TryLoadBinaryStream("TSModel", r => model.load(r)))
                throw env.ExceptDecode();
        }

        private protected void LoadModelHeader(IHostEnvironment env, ModelLoadContext ctx, VersionInfo versionInfo)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(versionInfo);
        }

        private protected void SaveModelHeader(ModelSaveContext ctx, VersionInfo versionInfo)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(versionInfo);
        }

        private protected void SaveTorchSharpModel(ModelSaveContext ctx)
        {
            // *** Binary format ***
            // Binary Stream of the model

            ctx.SaveBinaryStream("TSModel", w =>
            {
                Model.save(w);
            });
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => GetRowMapper(this, schema);

        private protected abstract IRowMapper GetRowMapper(TorchSharpTransformer parent, DataViewSchema schema);

        private protected abstract class Mapper : MapperBase
        {
            private readonly TorchSharpTransformer _parent;
            private readonly DataViewSchema _inputSchema;

            public Mapper(TorchSharpTransformer parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _inputSchema = inputSchema;

                torch.random.manual_seed(1);
                torch.cuda.manual_seed(1);
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
        }
    }
}
