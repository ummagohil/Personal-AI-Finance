// SavingsExpenditureAnalysis.tsx
import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

// Directly import the JSON data
import data from "./data/savings_and_current_account_data.json"; // Adjust the path as needed

import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  ChartData,
} from "chart.js";

// Registering components for Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const SavingsExpenditureAnalysis: React.FC = () => {
  const [chartData, setChartData] = useState<ChartData<"line">>({
    labels: [],
    datasets: [],
  });
  const [forecastData, setForecastData] = useState<{
    label: string;
    value: number | null;
  }>({
    label: "Forecasted Expenditure",
    value: null,
  });

  useEffect(() => {
    // Extract data for processing
    const savings: number[] = data.map(
      (entry) => entry.Savings_Account_Balance
    );
    const expenditure: number[] = data.map(
      (entry) => entry.Current_Account_Expenditure
    );

    // Normalize data
    const normalize = (values: number[]) => {
      const min = Math.min(...values);
      const max = Math.max(...values);
      return values.map((value) => (value - min) / (max - min));
    };

    const normalizedSavings = normalize(savings);
    const normalizedExpenditure = normalize(expenditure);

    // Prepare TensorFlow tensors
    const savingsTensor = tf.tensor1d(normalizedSavings);
    const expenditureTensor = tf.tensor1d(normalizedExpenditure);

    // Define and compile the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({
      optimizer: tf.train.sgd(0.001),
      loss: "meanSquaredError",
    });

    // Train the model
    model
      .fit(savingsTensor, expenditureTensor, {
        epochs: 100,
      })
      .then(() => {
        // Predict the values
        const predictionsTensor = model.predict(savingsTensor) as tf.Tensor;

        predictionsTensor.array().then((predictions: any) => {
          // Denormalize predictions
          const minExpenditure = Math.min(...expenditure);
          const maxExpenditure = Math.max(...expenditure);
          const denormalize = (value: number) =>
            value * (maxExpenditure - minExpenditure) + minExpenditure;
          const denormalizedPredictions = predictions.map((pred: any) =>
            denormalize(pred[0])
          );

          // Set chart data
          setChartData({
            labels: data.map((entry) =>
              new Date(entry.Date).toLocaleDateString()
            ),
            datasets: [
              {
                label: "Actual Expenditure",
                data: expenditure,
                borderColor: "#3b82f6", // Blue
                fill: false,
                tension: 0.3, // Smoother line
              },
              {
                label: "Predicted Expenditure",
                data: denormalizedPredictions,
                borderColor: "#ef4444", // Red
                borderDash: [5, 5],
                fill: false,
                tension: 0.3, // Smoother line
              },
            ],
          });

          // Forecast future values based on the last savings value
          const futureSavingsTensor = tf.tensor1d([
            normalizedSavings[normalizedSavings.length - 1],
          ]);
          const futurePredictions = model.predict(
            futureSavingsTensor
          ) as tf.Tensor;
          futurePredictions.array().then((forecast: any) => {
            const denormalizedForecast = denormalize(forecast[0][0]);

            // Update forecast data
            setForecastData({
              label: "Forecasted Expenditure",
              value: denormalizedForecast,
            });
          });
        });
      });
  }, []);

  // Chart options
  const options: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        ticks: {
          maxTicksLimit: 10, // Limit the number of ticks on the x-axis
        },
      },
      y: {
        beginAtZero: true, // Start the y-axis at zero for better readability
      },
    },
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Savings and Expenditure Over Time",
      },
      tooltip: {
        mode: "index",
        intersect: false, // Better tooltip visibility
      },
    },
  };

  return (
    <div className="p-8 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-center">
        Savings and Expenditure Dashboard
      </h1>

      <div className="bg-white p-6 rounded-lg shadow-md mb-8">
        <h2 className="text-2xl font-semibold mb-4">Expenditure Analysis</h2>
        <div className="relative h-[500px] w-full">
          {" "}
          {/* Adjusted height and width for better view */}
          <Line data={chartData} options={options} />
        </div>
      </div>

      {forecastData.value !== null && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-2xl font-semibold mb-4">
            Forecasted Expenditure
          </h2>
          <p className="text-lg">
            Forecasted Next Month Expenditure:{" "}
            <span className="font-bold text-red-500">
              {forecastData.value.toFixed(2)}
            </span>
          </p>
        </div>
      )}
    </div>
  );
};

export default SavingsExpenditureAnalysis;
