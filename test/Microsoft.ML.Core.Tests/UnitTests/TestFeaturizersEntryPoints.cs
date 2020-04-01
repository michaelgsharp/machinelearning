// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Core.Tests.UnitTests;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Featurizers;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.TestFrameworkCommon;
using Microsoft.ML.TestFrameworkCommon.Attributes;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Ensemble;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Transforms.TimeSeries;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;
using Xunit.Abstractions;

using Microsoft.ML.RunTests;

using Microsoft.ML.Tests.Transformers;
namespace Microsoft.ML.RunTests
{
    public class TestFeaturizersEntryPoints : CoreBaseTestClass
    {
        public TestFeaturizersEntryPoints(ITestOutputHelper output) : base(output)
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(RollingWindowEstimator).Assembly);
        }

        [Fact]
        public void AnalyticalRollingWindow_SimpleMeanTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.AnalyticalRollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'TargetColumn' : 'Target',
                            'Data' : '$data',
                            'Horizon': 1,
                            'MaxWindowSize' : 1,
                            'MinWindowSize' : 1,
                            'WindowCalculation' : 'Mean'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            var addedColumn = schema["Target_Mean_Hor1_MinWin1_MaxWin1"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 1);
            Assert.True(columnType.Dimensions[1] == 1);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            Done();
        }

        [Fact]
        public void AnalyticalRollingWindow_SimpleMinTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.AnalyticalRollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'TargetColumn' : 'Target',
                            'Data' : '$data',
                            'Horizon': 1,
                            'MaxWindowSize' : 1,
                            'MinWindowSize' : 1,
                            'WindowCalculation' : 'Min'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            var addedColumn = schema["Target_Min_Hor1_MinWin1_MaxWin1"];
            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN }, new[] { 1d }, new[] { 2d }, new[] { 3d } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);

            VBuffer<double> buffer = default;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);
            }
            // TODO: Uncomment when featurizer fixed.
            //TestEstimatorCore(pipeline, data);
            Done();
        }
        [Fact]
        public void AnalyticalRollingWindow_SimpleMaxTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.AnalyticalRollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'TargetColumn' : 'Target',
                            'Data' : '$data',
                            'Horizon': 1,
                            'MaxWindowSize' : 1,
                            'MinWindowSize' : 1,
                            'WindowCalculation' : 'Max'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            var addedColumn = schema["Target_Max_Hor1_MinWin1_MaxWin1"];
            var cursor = output.GetRowCursor(addedColumn);

            var expectedOutput = new[] { new[] { double.NaN }, new[] { 1d }, new[] { 2d }, new[] { 3d } };
            var index = 0;
            var getter = cursor.GetGetter<VBuffer<double>>(addedColumn);

            VBuffer<double> buffer = default;

            while (cursor.MoveNext())
            {
                getter(ref buffer);
                var bufferValues = buffer.GetValues();

                Assert.Equal(expectedOutput[index].Length, bufferValues.Length);
                Assert.Equal(expectedOutput[index++][0], bufferValues[0]);
            }

            // TODO: Uncomment when featurizer fixed.
            //TestEstimatorCore(pipeline, data);
            Done();
        }
        [Fact]
        public void AnalyticalRollingWindow_ComplexTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.AnalyticalRollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'TargetColumn' : 'Target',
                            'Data' : '$data',
                            'Horizon': 4,
                            'MaxWindowSize' : 3,
                            'MinWindowSize' : 2,
                            'WindowCalculation' : 'Mean'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            var addedColumn = schema["Target_Mean_Hor4_MinWin2_MaxWin3"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 1);
            Assert.True(columnType.Dimensions[1] == 4);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            Done();
        }

        [Fact]
        public void LagLead_EntryPointTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.LagLeadOperator',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'TargetColumn' : 'Target',
                            'Data' : '$data',
                            'Horizon': 1,
                            'offsets' : [-3, 1]
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            // TODO: complete this test after lag lead is fully implemented
            Done();
        }

        [Fact]
        public void ForecastingPivot_EntryPointTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['Grain'],
                            'Data' : '$data'
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            // TODO: complete this test after forecasting pivot is fully implemented

            Done();
        }
        [Fact]
        public void ShortDrop_EntryPointTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 },
                new { Grain = "one", Target = 2.0 },
                new { Grain = "one", Target = 3.0 },
                new { Grain = "one", Target = 4.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ShortDrop',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Horizon': 1,
                            'Data' : '$data',
                            'MaxWindowSize' : 3,
                            'CrossValidations' : 2,
                            'offsets' : [-3, 1]
                    },
                    'Outputs' : {
                        'OutputData': '$outputData',
                        'Model': '$Var_test'
                    }
                    }
                ]
            }
            ";

            JObject graph = JObject.Parse(inputGraph);

            var runner = new GraphRunner(Env, graph[FieldNames.Nodes] as JArray);

            // TODO: complete this test after short grain dropper is fully implemented

            Done();
        }
    }
}
