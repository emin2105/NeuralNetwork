using System.Collections;
using System.Collections.Generic;

public class Layer
{
    public int neuronsCount;
    public List<Neuron> neurons = new List<Neuron>();

    public Layer(int neuronsCount, int inputsCount)
    {
        this.neuronsCount = neuronsCount;
        for (int i = 0; i < neuronsCount; i++)
        {
            neurons.Add(new Neuron(inputsCount));
        }
    }
}