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


        private class DateTimeInput
        {
            public long date;
        }
        public TestFeaturizersEntryPoints(ITestOutputHelper output) : base(output)
        {
            Env.ComponentCatalog.RegisterAssembly(typeof(ExponentialAverageTransform).Assembly);
        }

        [Fact]
        public void DateTime()
        {
            // Date - 2025 June 30
            MLContext mlContext = new MLContext(1);
            var dataList = new[] { new DateTimeInput() { date = 1751241600 } };
            var data = mlContext.Data.LoadFromEnumerable(dataList);

            string inputGraph = @"
            {
                'Nodes':
                [{
                    'Name': 'Transforms.DateTimeSplitter',
                    'Inputs': {
                            'Source': 'date',
                            'Data' : 'data',
                            'Prefix' : 'pref_'
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

            // Get the data from the first row and make sure it matches expected
            var row = output.Preview(1).RowView[0].Values;
            /*
            // Assert the data from the first row is what we expect
            Assert.Equal(2025, row[1].Value);                           // Year
            Assert.Equal((byte)6, row[2].Value);                        // Month
            Assert.Equal((byte)30, row[3].Value);                       // Day
            Assert.Equal((byte)0, row[4].Value);                        // Hour
            Assert.Equal((byte)0, row[5].Value);                        // Minute
            Assert.Equal((byte)0, row[6].Value);                        // Second
            Assert.Equal((byte)0, row[7].Value);                        // AmPm
            Assert.Equal((byte)0, row[8].Value);                        // Hour12
            Assert.Equal((byte)1, row[9].Value);                        // DayOfWeek
            Assert.Equal((byte)91, row[10].Value);                      // DayOfQuarter
            Assert.Equal((ushort)180, row[11].Value);                   // DayOfYear
            Assert.Equal((ushort)4, row[12].Value);                     // WeekOfMonth
            Assert.Equal((byte)2, row[13].Value);                       // QuarterOfYear
            Assert.Equal((byte)1, row[14].Value);                       // HalfOfYear
            Assert.Equal((byte)27, row[15].Value);                      // WeekIso
            Assert.Equal(2025, row[16].Value);                          // YearIso
            Assert.Equal("June", row[17].Value.ToString());             // MonthLabel
            Assert.Equal("am", row[18].Value.ToString());               // AmPmLabel
            Assert.Equal("Monday", row[19].Value.ToString());           // DayOfWeekLabel
            Assert.Equal("", row[20].Value.ToString());  // HolidayName
            Assert.Equal((byte)0, row[21].Value);                       // IsPaidTimeOff

            Done();
            */
        }
    }
}
