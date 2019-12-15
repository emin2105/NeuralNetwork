using System;
using System.Collections;
using System.Collections.Generic;

public class NeuralNetwork
{
    public int inputsCount;
    public int outputsCount;
    public int hiddenLayersCount;
    public int neuronsPerHidden;
    public double alpha;
    public List<Layer> layers = new List<Layer>();

    public NeuralNetwork(int inputsCount, int outputsCount, int hiddenLayersCount, int neuronsPerHidden, double alpha)
    {
        this.inputsCount = inputsCount;
        this.outputsCount = outputsCount;
        this.hiddenLayersCount = hiddenLayersCount;
        this.neuronsPerHidden = neuronsPerHidden;
        this.alpha = alpha;

        if (hiddenLayersCount > 0)
        {
            layers.Add(new Layer(this.neuronsPerHidden, inputsCount));
            for (int i = 1; i < hiddenLayersCount - 1; i++)
            {
                layers.Add(new Layer(this.neuronsPerHidden, this.neuronsPerHidden));
            }
            layers.Add(new Layer(this.outputsCount, this.neuronsPerHidden));
        }
        else
        {
            layers.Add(new Layer(this.outputsCount, inputsCount));
        }
    }

    public List<double> Go(List<double> inputValues, List<double> desiredOutputs, bool updateWeights = true)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if (inputValues.Count != inputsCount)
        {
            throw new ArgumentException($"Number of inputs must be equal to {this.inputsCount} in tnis nn");
        }
        inputs = new List<double>(inputValues);
        for (int i = 0; i <= hiddenLayersCount; i++)
        {
            if (i > 0)
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            for (int j = 0; j < layers[i].neuronsCount; j++)
            {
                double N = 0;
                layers[i].neurons[j].inputs.Clear();
                for (int k = 0; k < layers[i].neurons[j].inputsCount; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    N += layers[i].neurons[j].weights[k] * inputs[k];
                }
                N -= layers[i].neurons[j].bias;
                layers[i].neurons[j].output = ActivationFunction(N);
                outputs.Add(layers[i].neurons[j].output);
            }

        }
        if (updateWeights)
            UpdateWeights(outputs, desiredOutputs);
        return outputs;
    }

    double ActivationFunction(double inputSum)
    {
        return Sigmoid(inputSum);
    }
    double Step(double value)
    {
        return value < 0 ? 0 : 1;
    }
    double Sigmoid(double value)
    {
        double k = (double) System.Math.Exp(value);
        return k / (1.0f + k);
    }
    void UpdateWeights(List<double> outputs, List<double> desiredOutputs)
    {
        double error;
        for (int i = hiddenLayersCount; i >= 0; i--)
        {
            for (int j = 0; j < layers[i].neuronsCount; j++)
            {
                if (i == hiddenLayersCount)
                {
                    error = desiredOutputs[j] - outputs[j];
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                }
                else
                {
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output);
                    double errorGradientSum = 0;
                    for (int p = 0; p < layers[i + 1].neuronsCount; p++)
                    {
                        errorGradientSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    layers[i].neurons[j].errorGradient *= errorGradientSum;
                }
                for (int k = 0; k < layers[i].neurons[j].inputsCount; k++)
                {
                    if (i == hiddenLayersCount)
                    {

                        error = desiredOutputs[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {

                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                    }
                }

                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient;
            }
        }
    }


}
