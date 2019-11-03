using Microsoft.ML.Data;

namespace AI_esame
{
    class Prediction
    {
        // Vector to hold Alert, Score, and P-Value values
        [VectorType(3)]
        public double[] Values { get; set; }
    }
}
