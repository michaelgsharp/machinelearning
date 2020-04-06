using Microsoft.ML.Data;
using Microsoft.ML.Featurizers;
using Microsoft.ML.RunTests;
using System;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;
using System.Linq;
using Microsoft.ML.TestFramework.Attributes;

namespace Microsoft.ML.Tests.Transformers
{
    public class ForecastingPivotFeaturizerTests : TestDataPipeBase
    {
        public ForecastingPivotFeaturizerTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestInvalidType()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { ColA_Mean_MinWin1_MaxWin1 = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline, should error on fit and on GetOutputSchema
            var pipeline = mlContext.Transforms.PivotForecastingData(new string[] { "ColA_Mean_MinWin1_MaxWin1" });

            Assert.Throws<InvalidOperationException>(() => pipeline.Fit(data));
            Assert.Throws<InvalidOperationException>(() => pipeline.GetOutputSchema(SchemaShape.Create(data.Schema)));

            Done();
        }

        [Fact]
        public void SimpleSchemaTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 }
            };

            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline. Starting with RollingWindow since this depends on RollingWindow or LagLead.
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1).Append(
                mlContext.Transforms.PivotForecastingData(new string[] { "ColA_Mean_MinWin1_MaxWin1" })
            );
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            var addedColumn = schema["ColA_Mean_MinWin1_MaxWin1"];
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
        public void SimpleTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 }
            };

            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 1, 1).Append(
                mlContext.Transforms.PivotForecastingData(new string[] { "ColA_Mean_MinWin1_MaxWin1" })
            );
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var pivotCol = debugView.ColumnView[3].Values;
            var horizonCol = debugView.ColumnView[4].Values;

            // Length should be 2 because we drop the first row.
            Assert.True(pivotCol.Length == 2);

            // Make sure the values are correct.
            Assert.Equal(1.0, pivotCol[0]);
            Assert.Equal(2.0, pivotCol[1]);

            Assert.Equal((UInt32)1, horizonCol[0]);
            Assert.Equal((UInt32)1, horizonCol[1]);

            Done();
        }

        [Fact]
        public void Horizon2Test()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 },
                new { GrainA = "Grain", ColA = 2.0 },
                new { GrainA = "Grain", ColA = 3.0 }
            };

            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Build the pipeline
            var pipeline = mlContext.Transforms.RollingWindow(new string[] { "GrainA" }, "ColA", RollingWindowEstimator.RollingWindowCalculation.Mean, 2, 1).Append(
                mlContext.Transforms.PivotForecastingData(new string[] { "ColA_Mean_MinWin1_MaxWin1" })
            );
            var model = pipeline.Fit(data);
            var output = model.Transform(data);
            var schema = output.Schema;

            //var index = 0;
            var debugView = output.Preview();
            var colA = debugView.ColumnView[1].Values;
            var pivotCol = debugView.ColumnView[3].Values;
            var horizonCol = debugView.ColumnView[4].Values;

            // Correct output for:
            // ColA,    ColA_Mean_MinWin1_MaxWin1,  Horizon
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
        public void ConstructorParameterTest()
        {
            MLContext mlContext = new MLContext(1);
            var dataList = new[] {
                new { GrainA = "Grain", ColA = 1.0 }
            };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            // Make sure invalid constructor args throw.
            Assert.Throws<ArgumentOutOfRangeException>(() => mlContext.Transforms.PivotForecastingData(new string[] { }));
            Assert.Throws<ArgumentNullException>(() => mlContext.Transforms.PivotForecastingData(null));

            Done();
        }

    }
}
