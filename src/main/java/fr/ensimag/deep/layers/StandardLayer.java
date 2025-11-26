package fr.ensimag.deep.layers;

import org.ejml.simple.SimpleMatrix;

import fr.ensimag.deep.activators.IActivator;

public class StandardLayer extends AbstractLayer{

    public StandardLayer(SimpleMatrix initialWeights, SimpleMatrix initialBias, IActivator activator)
    {
        super(initialBias.getNumRows(), initialWeights.getNumRows(), activator);
    }

    @Override
    public void propagate(SimpleMatrix input) {

        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public SimpleMatrix getActivation() {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public SimpleMatrix getWeightedError() {
        throw new UnsupportedOperationException("not yet implemented");
    }    

    @Override
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        // to be completed
        for(int i=0,i<batchSize,i++){
            
        }

    }

    @Override
    public void updateParameters() {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public void backpropagate(SimpleMatrix upperWeightedError) {
        throw new UnsupportedOperationException("not yet implemented");
    }
                

    @Override
    public SimpleMatrix getWeights() {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public SimpleMatrix getBias() {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public void setWeights(SimpleMatrix weights) {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public void setBias(SimpleMatrix bias) {
        throw new UnsupportedOperationException("not yet implemented");
    }
}
