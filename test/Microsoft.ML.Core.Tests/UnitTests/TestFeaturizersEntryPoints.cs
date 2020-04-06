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
        private class SimpleRWTestData
        {
            public double ColA { get; set; }

            [VectorType(1, 2)]
            public double[] ColA_RW_Mean_MinWin1_MaxWin1 { get; set; }
        }
        private class SimpleLagLeadTestData
        {
            public double ColA { get; set; }

            [VectorType(2, 2)]
            public double[] ColA_Lag_1_Lead_1 { get; set; }
        }

        private class Horizon2LagLeadRWTestData
        {
            public double ColA { get; set; }

            [VectorType(1, 2)]
            public double[] ColA_RW_Mean_MinWin1_MaxWin1 { get; set; }

            [VectorType(2, 2)]
            public double[] ColA_Lag_1_Lead_1 { get; set; }
        }

        [Fact]
        public void AnalyticalRollingWindow_LargeNumberTest()
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
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Column' : { 'Name' : 'Target_RW_Min_MinWin4294967294_MaxWin4294967294', 'Source' : 'Target' },
                            'Data' : '$data',
                            'Horizon': 2147483647,
                            'MaxWindowSize' : 4294967294,
                            'MinWindowSize' : 4294967294,
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

            var addedColumn = schema["Target_RW_Min_MinWin4294967294_MaxWin4294967294"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 1);
            Assert.True(columnType.Dimensions[1] == 2147483647);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;

            annotations.GetValue<ReadOnlyMemory<char>>("FeaturizerName", ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>("Calculation", ref calculation);
            annotations.GetValue<UInt32>("MinWindowSize", ref minWindowSize);
            annotations.GetValue<UInt32>("MaxWindowSize", ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Min", calculation.ToString());
            Assert.Equal((UInt32)4294967294, minWindowSize);
            Assert.Equal((UInt32)4294967294, maxWindowSize);
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
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Column' : { 'Name' : 'Target_RW_Mean_MinWin1_MaxWin1', 'Source' : 'Target' },
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

            var addedColumn = schema["Target_RW_Mean_MinWin1_MaxWin1"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 1);
            Assert.True(columnType.Dimensions[1] == 1);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;

            annotations.GetValue<ReadOnlyMemory<char>>("FeaturizerName", ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>("Calculation", ref calculation);
            annotations.GetValue<UInt32>("MinWindowSize", ref minWindowSize);
            annotations.GetValue<UInt32>("MaxWindowSize", ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Mean", calculation.ToString());
            Assert.Equal((UInt32)1, minWindowSize);
            Assert.Equal((UInt32)1, maxWindowSize);

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
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Column' : { 'Name' : 'Target_RW_Min_MinWin1_MaxWin1', 'Source' : 'Target' },
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

            var addedColumn = schema["Target_RW_Min_MinWin1_MaxWin1"];
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

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;

            annotations.GetValue<ReadOnlyMemory<char>>("FeaturizerName", ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>("Calculation", ref calculation);
            annotations.GetValue<UInt32>("MinWindowSize", ref minWindowSize);
            annotations.GetValue<UInt32>("MaxWindowSize", ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Min", calculation.ToString());
            Assert.Equal((UInt32)1, minWindowSize);
            Assert.Equal((UInt32)1, maxWindowSize);

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
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Column' : { 'Name' : 'Target_RW_Max_MinWin1_MaxWin1', 'Source' : 'Target' },
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

            var addedColumn = schema["Target_RW_Max_MinWin1_MaxWin1"];
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

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;

            annotations.GetValue<ReadOnlyMemory<char>>("FeaturizerName", ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>("Calculation", ref calculation);
            annotations.GetValue<UInt32>("MinWindowSize", ref minWindowSize);
            annotations.GetValue<UInt32>("MaxWindowSize", ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Max", calculation.ToString());
            Assert.Equal((UInt32)1, minWindowSize);
            Assert.Equal((UInt32)1, maxWindowSize);

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
                    'Name': 'Transforms.RollingWindow',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Column' : { 'Name' : 'Target_RW_Mean_MinWin2_MaxWin3', 'Source' : 'Target' },
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

            var addedColumn = schema["Target_RW_Mean_MinWin2_MaxWin3"];
            var columnType = addedColumn.Type as VectorDataViewType;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType.IsKnownSize);
            Assert.True(columnType.Dimensions.Length == 2);
            Assert.True(columnType.Dimensions[0] == 1);
            Assert.True(columnType.Dimensions[1] == 4);
            Assert.True(columnType.ItemType.RawType == typeof(double));

            // Verify annotations are correct.
            ReadOnlyMemory<char> featurizerName = default;
            ReadOnlyMemory<char> calculation = default;
            UInt32 minWindowSize = default;
            UInt32 maxWindowSize = default;

            var annotations = addedColumn.Annotations;

            annotations.GetValue<ReadOnlyMemory<char>>("FeaturizerName", ref featurizerName);
            annotations.GetValue<ReadOnlyMemory<char>>("Calculation", ref calculation);
            annotations.GetValue<UInt32>("MinWindowSize", ref minWindowSize);
            annotations.GetValue<UInt32>("MaxWindowSize", ref maxWindowSize);

            Assert.Equal("RollingWindow", featurizerName.ToString());
            Assert.Equal("Mean", calculation.ToString());
            Assert.Equal((UInt32)2, minWindowSize);
            Assert.Equal((UInt32)3, maxWindowSize);

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
                            'Offsets' : [-3, 1]
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
            // TODO: complete this test after lag lead is fully implemented
            Done();
        }

        [Fact]
        public void ForecastingPivot_SimpleRWTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleRWTestData { ColA = 1.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { double.NaN, double.NaN } },
                new SimpleRWTestData { ColA = 2.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { double.NaN, 1.0 } },
                new SimpleRWTestData { ColA = 3.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { 1.0, 2.0 } }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['ColA_RW_Mean_MinWin1_MaxWin1'],
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

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            var addedColumn = schema["ColA_RW_Mean_MinWin1_MaxWin1"];
            var columnType = addedColumn.Type;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType == NumberDataViewType.Double);

            addedColumn = schema["Horizon"];
            columnType = addedColumn.Type;

            // Make sure the type and schema of the column are correct.
            Assert.NotNull(columnType);
            Assert.True(columnType == NumberDataViewType.UInt32);

            Done();
        }

        [Fact]
        public void ForecastingPivot_SimpleLagLeadTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleLagLeadTestData { ColA = 1.0, ColA_Lag_1_Lead_1 = new [] { double.NaN, double.NaN, double.NaN, double.NaN } },
                new SimpleLagLeadTestData { ColA = 2.0, ColA_Lag_1_Lead_1 = new [] { double.NaN, 1.0, double.NaN, 1.0 } },
                new SimpleLagLeadTestData { ColA = 3.0, ColA_Lag_1_Lead_1 = new [] { 1.0, 2.0, 1.0, 2.0 } }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['ColA_Lag_1_Lead_1'],
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

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[0].Values;
            var lagCol = debugView.ColumnView[2].Values;
            var leadCol = debugView.ColumnView[3].Values;
            var horizonCol = debugView.ColumnView[4].Values;

            // Correct output for:
            // ColA,    ColA_Lag_1, ColA_Lead_1,    Horizon
            // 2.0,     1.0,        1.0,            1
            // 3.0,     1.0,        1.0,            2
            // 3.0,     2.0,        2.0,            1

            Assert.True(leadCol.Length == 3);

            // Make sure the values are correct.
            Assert.Equal(2.0, colA[0]);
            Assert.Equal(3.0, colA[1]);
            Assert.Equal(3.0, colA[2]);

            Assert.Equal(1.0, leadCol[0]);
            Assert.Equal(1.0, leadCol[1]);
            Assert.Equal(2.0, leadCol[2]);

            Assert.Equal(1.0, lagCol[0]);
            Assert.Equal(1.0, lagCol[1]);
            Assert.Equal(2.0, lagCol[2]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)2, horizonCol[1]);
            Assert.Equal((UInt32)1, horizonCol[2]);

            Done();
        }
        [Fact]
        public void ForecastingPivot_Horizon2RWTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new SimpleRWTestData { ColA = 1.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { double.NaN, double.NaN } },
                new SimpleRWTestData { ColA = 2.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { double.NaN, 1.0 } },
                new SimpleRWTestData { ColA = 3.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { 1.0, 2.0 } }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['ColA_RW_Mean_MinWin1_MaxWin1'],
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

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[0].Values;
            var pivotCol = debugView.ColumnView[2].Values;
            var horizonCol = debugView.ColumnView[3].Values;

            // Correct output for:
            // ColA,    ColA_RW_Mean_MinWin1_MaxWin1,  Horizon
            // 2.0,     1.0,                        1
            // 3.0,     1.0,                        2
            // 3.0,     2.0,                        1

            Assert.True(pivotCol.Length == 3);

            // Make sure the values are correct.
            Assert.Equal(2.0, colA[0]);
            Assert.Equal(3.0, colA[1]);
            Assert.Equal(3.0, colA[2]);

            Assert.Equal(1.0, pivotCol[0]);
            Assert.Equal(1.0, pivotCol[1]);
            Assert.Equal(2.0, pivotCol[2]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)2, horizonCol[1]);
            Assert.Equal((UInt32)1, horizonCol[2]);

            Done();
        }

        [Fact]
        public void ForecastingPivot_Horizon2LagLeadTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new Horizon2LagLeadRWTestData { ColA = 1.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { double.NaN, double.NaN }, ColA_Lag_1_Lead_1 = new [] { double.NaN, double.NaN, double.NaN, double.NaN } },
                new Horizon2LagLeadRWTestData { ColA = 2.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { double.NaN, 1.0 }, ColA_Lag_1_Lead_1 = new [] { double.NaN, 1.0, double.NaN, 2.0 } },
                new Horizon2LagLeadRWTestData { ColA = 3.0, ColA_RW_Mean_MinWin1_MaxWin1 = new [] { 1.0, 2.0 }, ColA_Lag_1_Lead_1 = new [] { 2.0, double.NaN, 3.0, double.NaN } }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ForecastingPivot',
                    'Inputs': {
                            'ColumnsToPivot': ['ColA_RW_Mean_MinWin1_MaxWin1', 'ColA_Lag_1_Lead_1'],
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

            runner.SetInput("data", data);
            runner.RunAll();

            var output = runner.GetOutput<IDataView>("outputData");
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[0].Values;
            var rollingWindowCol = debugView.ColumnView[3].Values;
            var lagCol = debugView.ColumnView[4].Values;
            var leadCol = debugView.ColumnView[5].Values;
            var horizonCol = debugView.ColumnView[6].Values;

            // Correct output for:
            // ColA,    ColA_RW_Mean_MinWin1_MaxWin1,  ColA_Lag_1, ColA_Lead_1,    Horizon
            // 2.0,     1.0,                        1.0,        2.0,            1
            // 3.0,     1.0,                        2.0,        3.0,            2

            Assert.True(colA.Length == 2);

            // Make sure the values are correct.
            Assert.Equal(2.0, colA[0]);
            Assert.Equal(3.0, colA[1]);

            Assert.Equal(1.0, rollingWindowCol[0]);
            Assert.Equal(1.0, rollingWindowCol[1]);

            Assert.Equal(1.0, lagCol[0]);
            Assert.Equal(2.0, lagCol[1]);

            Assert.Equal(2.0, leadCol[0]);
            Assert.Equal(3.0, leadCol[1]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)2, horizonCol[1]);

            Done();
        }

        [Fact]
        public void ShortDrop_LargeNumberTest()
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
                            'Data' : '$data',
                            'MinPoints' : 4294967294
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

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 4294967294 and we only have 1, should have no rows back.
            Assert.True(rows.Length == 0);

            // Also make sure that the value column was dropped correctly.
            Assert.True(cols[0].Values.Length == 0);
            Assert.True(cols[1].Values.Length == 0);

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
                            'Data' : '$data',
                            'MinPoints' : 2
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

            Done();
        }
        [Fact]
        public void ShortDrop_Drop()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ShortDrop',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Data' : '$data',
                            'MinPoints' : 2
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

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 2 and we only have 1, should have no rows back.
            Assert.True(rows.Length == 0);

            // Also make sure that the value column was dropped correctly.
            Assert.True(cols[0].Values.Length == 0);
            Assert.True(cols[1].Values.Length == 0);

            Done();
        }
        [Fact]
        public void ShortDrop_Keep()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { Grain = "one", Target = 1 },
                new { Grain = "one", Target = 1 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.ShortDrop',
                    'Inputs': {
                            'GrainColumns': ['Grain'],
                            'Data' : '$data',
                            'MinPoints' : 2
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

            // Output schema should equal input schema.
            Assert.Equal(data.Schema, schema);
            var debugView = output.Preview();
            var rows = debugView.RowView;
            var cols = debugView.ColumnView;

            // Since min row count is 2 and we have 2, should have all rows back.
            Assert.True(rows.Length == 2);

            // Also make sure that the value column was dropped correctly.
            Assert.True(cols[0].Values.Length == 2);
            Assert.True(cols[1].Values.Length == 2);

            Done();
        }
    }
}
