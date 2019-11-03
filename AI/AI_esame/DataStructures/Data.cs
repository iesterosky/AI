using Microsoft.ML.Data;

namespace AI_esame
{
    public class Data
    {
        [LoadColumn(0)]
        public string label;

        [LoadColumn(1)]
        public float data_value;

        [LoadColumn(3)]
        public double p_value;

        [LoadColumn(3)]
        public double martingale_value;

        [LoadColumn(3)]
        public bool alertSpike;

        [LoadColumn(3)]
        public bool alertChange;
    }
}
