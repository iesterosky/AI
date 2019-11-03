using System;
using Microsoft.ML;
using System.IO;
using System.Collections.Generic;

namespace AI_esame
{
    class Program
    {
        private static string DatasetPath;
        private static MLContext mlContext;

        static void Main()
        {
            Console.WriteLine("==== Inserisci il percorso del file .csv da analizzare ====");
            Console.Write("Percorso: ");
            DatasetPath = Console.ReadLine();

            while (!File.Exists(DatasetPath))
            {
                Console.WriteLine("==== Percorso errato. Inserisci il percorso del file .csv da analizzare ====");
                DatasetPath = Console.ReadLine();
            }

            // Creo l'oggetto MLContext 
            mlContext = new MLContext();
            
            // Carico i dati nell'IDataView considerando la classe personalizzata Data come contenitore di dati.
            IDataView dataView = mlContext.Data.LoadFromTextFile<Data>(path: DatasetPath, hasHeader: true, separatorChar: ',');
            
            var loadedDataEnumerable = mlContext.Data.CreateEnumerable<Data>(dataView, reuseRowObject: false);

            //Salvo i dati in una lista per una facile manipolazione e conto il numero di record
            List<Data> data = new List<Data>();

            int size = 0;
            foreach (Data row in loadedDataEnumerable)
            {
                size++;
                data.Add(row);
            }

            // Trovo gli spikes nel pattern, ovvero i cambiamenti temporanei
            IEnumerable<Prediction> spikePredictions = DetectSpike(size, dataView);
            //Aggiorno la lista precedentemente salvata con le nuove informazioni
            saveSpikePredictions(ref data, spikePredictions);

            // Trovo i cambiamenti persistenti nel pattern
            IEnumerable<Prediction> changePredictions = DetectChangepoint(size, dataView);
            //Aggiorno la lista precedentemente salvata con le nuove informazioni
            saveChangePredictions(ref data, changePredictions);

            //Posso stampare i dati ottenuti
            printData(data);

            Console.WriteLine("==== Tutto completato! E' stato salvato il file \"report.txt\" con i risultati. ====\n==== Premere INVIO per terminare ====");
            Console.ReadLine();
        }

        private static IEnumerable<Prediction> DetectSpike(int size, IDataView dataView)
        {
            // STEP 1: Create Esimator.
            var estimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(Prediction.Values), inputColumnName: nameof(Data.data_value), confidence: 95, pvalueHistoryLength: size / 4);

            // STEP 2:The Transformed Model.
            // In IID Spike detection, we don't need to do training, we just need to do transformation. 
            // As you are not training the model, there is no need to load IDataView with real data, you just need schema of data.
            // So create empty data view and pass to Fit() method. 
            ITransformer tansformedModel = estimator.Fit(CreateEmptyDataView());

            // STEP 3: Use/test model.
            // Apply data transformation to create predictions.
            IDataView transformedData = tansformedModel.Transform(dataView);
           
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedData, reuseRowObject: false);

            return predictions;
        }

        private static void saveSpikePredictions(ref List<Data> data, IEnumerable<Prediction> spikePredictions)
        {
            IEnumerator<Prediction> predictions = spikePredictions.GetEnumerator();

            foreach (Data row in data)
            {
                predictions.MoveNext();
                Prediction prediction = predictions.Current;
                

                if (row.data_value != prediction.Values[1]) throw new Exception("Errore salvataggio predizioni spikes");

                row.alertSpike = prediction.Values[0] == 1;
                row.p_value = prediction.Values[2];
            }
            
        }

        private static IEnumerable<Prediction> DetectChangepoint(int size, IDataView dataView)
        {
            // STEP 1: Setup transformations using DetectIidChangePoint.
            var estimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(Prediction.Values), inputColumnName: nameof(Data.data_value), confidence: 95, changeHistoryLength: size / 4);

            // STEP 2:The Transformed Model.
            // In IID Change point detection, we don't need need to do training, we just need to do transformation. 
            // As you are not training the model, there is no need to load IDataView with real data, you just need schema of data.
            // So create empty data view and pass to Fit() method.  
            ITransformer tansformedModel = estimator.Fit(CreateEmptyDataView());

            // STEP 3: Use/test model.
            // Apply data transformation to create predictions.
            IDataView transformedData = tansformedModel.Transform(dataView);
            var predictions = mlContext.Data.CreateEnumerable<Prediction>(transformedData, reuseRowObject: false);
            
            return predictions;
        }


        private static void saveChangePredictions(ref List<Data> data, IEnumerable<Prediction> changePredictions)
        {
            IEnumerator<Prediction> predictions = changePredictions.GetEnumerator();
            foreach (Data row in data)
            {
                predictions.MoveNext();
                Prediction prediction = predictions.Current;

                if (row.data_value != prediction.Values[1]) throw new Exception("Errore salvataggio predizioni changes");

                row.alertChange = prediction.Values[0] == 1;
                row.martingale_value = prediction.Values[3];
            }

        }


        private static void printData(List<Data> data)
        {

            System.IO.StreamWriter file = new System.IO.StreamWriter(@"../../../../report.txt") ;

            file.WriteLine("Etichetta\tValore\t\tSpike\tChangePoint\tp-value\tmartingale-value");

            foreach (Data row in data)
            {
                file.WriteLine("{0}\t\t{1:0.00}\t\t{2}\t{3}\t\t{4:0.00}\t{5:0.00}", row.label, row.data_value, row.alertSpike, row.alertChange, row.p_value, row.martingale_value);
            }

            file.Close();
        }

        private static IDataView CreateEmptyDataView()
        {
            //Create empty DataView. We just need the schema to call fit()
            IEnumerable<Data> enumerableData = new List<Data>();
            var dv = mlContext.Data.LoadFromEnumerable(enumerableData);
            return dv;
        }
    }
}