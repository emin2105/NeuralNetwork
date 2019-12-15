using System;
using System.Collections;
using System.Collections.Generic;

public class Neuron
{
    public int inputsCount;
    public double bias;
    public double output;
    public double errorGradient;
    public List<double> weights = new List<double>();
    public List<double> inputs = new List<double>();

    public Neuron(int inputsCount)
    {
        var random = new Random();
        for (int i = 0; i < inputsCount; i++)
        {
            bias = random.NextDouble() * 2 - 1;
            this.inputsCount = inputsCount;
            weights.Add(random.NextDouble() * 2 - 1);
        }
    }

}
