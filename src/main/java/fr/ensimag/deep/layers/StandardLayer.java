package fr.ensimag.deep.layers;

import org.ejml.simple.SimpleMatrix;
import fr.ensimag.deep.activators.IActivator;

public class StandardLayer extends AbstractLayer {

    private SimpleMatrix weights;      // Taille: previousLayerSize × layerSize
    private SimpleMatrix bias;         // Taille: layerSize × 1
    private SimpleMatrix activation;   // Taille: layerSize × batchSize (résultat après φ)
    private SimpleMatrix weightedSum;  // Taille: layerSize × batchSize (z = W^T * input + bias)

    public StandardLayer(SimpleMatrix initialWeights, SimpleMatrix initialBias, IActivator activator) {
        super(initialBias.getNumRows(), initialWeights.getNumRows(), activator);
        this.weights = initialWeights;
        this.bias = initialBias;
    }

    @Override
    public void propagate(SimpleMatrix input) {
        // z = W^T × input + bias (bias broadcasté sur chaque colonne)
        // weights: (previousLayerSize × layerSize)
        // input: (previousLayerSize × batchSize)
        // W^T × input: (layerSize × batchSize)
        
        this.weightedSum = weights.transpose().mult(input);
        
        // Ajouter le bias à chaque colonne
        for (int col = 0; col < weightedSum.getNumCols(); col++) {
            for (int row = 0; row < weightedSum.getNumRows(); row++) {
                double val = weightedSum.get(row, col) + bias.get(row, 0);
                weightedSum.set(row, col, val);
            }
        }
        
        // Appliquer la fonction d'activation φ élément par élément
        this.activation = new SimpleMatrix(weightedSum.getNumRows(), weightedSum.getNumCols());
        for (int row = 0; row < weightedSum.getNumRows(); row++) {
            for (int col = 0; col < weightedSum.getNumCols(); col++) {
                double val = activator.phi(weightedSum.get(row, col));
                activation.set(row, col, val);
            }
        }
    }

    @Override
    public SimpleMatrix getActivation() {
        return this.activation;
    }

    @Override
    public SimpleMatrix getWeights() {
        return this.weights;
    }

    @Override
    public SimpleMatrix getBias() {
        return this.bias;
    }

    @Override
    public void setWeights(SimpleMatrix weights) {
        this.weights = weights;
    }

    @Override
    public void setBias(SimpleMatrix bias) {
        this.bias = bias;
    }

    @Override
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    // Ces méthodes sont pour la backpropagation (Hands-on 2)
    @Override
    public SimpleMatrix getWeightedError() {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public void updateParameters() {
        throw new UnsupportedOperationException("not yet implemented");
    }

    @Override
    public void backpropagate(SimpleMatrix upperWeightedError) {
        throw new UnsupportedOperationException("not yet implemented");
    }
}