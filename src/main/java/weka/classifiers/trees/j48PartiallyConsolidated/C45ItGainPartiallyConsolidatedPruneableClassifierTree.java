package weka.classifiers.trees.j48PartiallyConsolidated;

import java.util.ArrayList;

import weka.classifiers.trees.J48PartiallyConsolidated;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a 'partial' consolidated tree structure,
 * based on a consolidation percentage or on a specific value,
 * that indicates the number of inner nodes of the tree to be developed.
 * The tree is processed iteratively (instead of recursively like 
 * the original version) and allows to decide which will be the 
 * next node to be developed according to a priority criterion;
 * in this case the gain ratio, the same criterion (split function) 
 * used by the C4.5 algorithm (J48) to choose the best attribute to 
 * split the node.
 * Up to 4 variants have been implemented:
 * 1) Gain ratio value associated with the training sample data at the current node.
 *  (m_priorityCriteria == PriorCrit_GainratioWholeData)
 * 2) Gain ratio value associated to the data of the vector of samples used in the 
 *  consolidated tree by calculating the mean of the contingency tables (distributions) 
 *  associated to each sample.
 *  (m_priorityCriteria == PriorCrit_GainratioSetSamples)
 * 3) Gain ratio (Whole data) multiplied by the node size.
 *  (m_priorityCriteria == PriorCrit_GainratioWholeData_Size)
 * 4) Gain ratio (Set of samples) multiplied by the node size.
 *  (m_priorityCriteria == PriorCrit_GainratioSetSamples_Size)
 *
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @author Josué Cabezas Regoyo
 * @version $Revision: 1.0 $
 */
public class C45ItGainPartiallyConsolidatedPruneableClassifierTree
		extends C45ItPartiallyConsolidatedPruneableClassifierTree {

	private static final long serialVersionUID = -7182722875887777854L;

	public C45ItGainPartiallyConsolidatedPruneableClassifierTree(ModelSelection toSelectLocModel,
			C45ModelSelectionExtended baseModelToForceDecision, boolean pruneTree, float cf, boolean raiseTree,
			boolean cleanup, boolean collapseTree, int numberSamples, int numberConsoNodesHowToSet,
			int priorityCriteria, int heuristicSearchAlgorithm, boolean pruneCT, boolean collapseCT,
			boolean notPreservingStructure) throws Exception {
		super(toSelectLocModel, baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree,
				numberSamples, numberConsoNodesHowToSet, priorityCriteria, heuristicSearchAlgorithm, pruneCT,
				collapseCT, notPreservingStructure);
	}

	/**
	 * Builds the partial consolidated tree structure, in this case
	 * iteratively (instead of recursively as in the original method, buildTree()).
	 * (m_priorityCriteria == PriorCrit_GainratioWholeData)
	 * (m_priorityCriteria == PriorCrit_GainratioSetSamples)
	 * (m_priorityCriteria == PriorCrit_GainratioWholeData_Size)
	 * (m_priorityCriteria == PriorCrit_GainratioSetSamples_Size)
	 *
	 * @param data          the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples used for consolidation
	 * @param keepData      is training data to be kept?
	 * @throws Exception if something goes wrong
	 */
	public void buildPartialTreeItera(Instances data, Instances[] samplesVector, boolean keepData) throws Exception {
		int numberSamples = samplesVector.length;
		Instances currentData;
		Instances[] currentSamplesVector;
		C45ItPartiallyConsolidatedPruneableClassifierTree currentTree;
		int index = 0;
		double orderValue;
		int internalNodes = 0;

		/** Initialize the consolidated tree */
		initiliazeTree(data, keepData);
		/** Initialize the base trees */
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample].initiliazeTree(samplesVector[iSample], keepData);

		/** List of nodes to be processed */
		ArrayList<Object[]> list = new ArrayList<>();

		// add(Data, samplesVector, tree, orderValue)
		list.add(new Object[]{data, samplesVector, this, null});

		while (list.size() > 0) {

			Object[] current = list.get(0);
			list.set(0, null); // Null to free up memory
			list.remove(0);

			currentData = (Instances)current[0];
			currentSamplesVector = (Instances[])current[1];
			currentTree = (C45ItPartiallyConsolidatedPruneableClassifierTree)current[2];

			currentTree.m_order = index;

			/** Initialize the consolidated tree */
			currentTree.initiliazeTree(currentData, keepData);
			/** Initialize the base trees */
			for (int iSample = 0; iSample < numberSamples; iSample++)
				currentTree.m_sampleTreeVector[iSample].initiliazeTree(currentSamplesVector[iSample], keepData);

			/**
			 * Select the best model to split (if it is worth) based on the consolidation
			 * proccess
			 */
			currentTree.setLocalModel(currentData, currentSamplesVector);
			for (int iSample = 0; iSample < numberSamples; iSample++)
				currentTree.m_sampleTreeVector[iSample].setLocalModel(currentSamplesVector[iSample],
						currentTree.getLocalModel());

			if ((currentTree.getLocalModel().numSubsets() > 1) && (internalNodes < m_maximumCriteria)) {

				/** Vector storing the obtained subsamples after the split of data */
				Instances[] localInstances;
				/**
				 * Vector storing the obtained subsamples after the split of each sample of the
				 * vector
				 */
				ArrayList<Instances[]> localInstancesVector = new ArrayList<Instances[]>();

				/**
				 * For some base trees, although the current node is not a leaf, it could be
				 * empty. This is necessary in order to calculate correctly the class membership
				 * probabilities for the given test instance in each base tree
				 */
				ArrayList<Object[]> listSons = new ArrayList<>();

				for (int iSample = 0; iSample < numberSamples; iSample++)
					if (Utils.eq(currentTree.m_sampleTreeVector[iSample].getLocalModel().distribution().total(), 0))
						currentTree.m_sampleTreeVector[iSample].setIsEmpty(true);

				/** Split data according to the consolidated m_localModel */
				localInstances = currentTree.getLocalModel().split(currentData);
				for (int iSample = 0; iSample < numberSamples; iSample++)
					localInstancesVector.add(currentTree.getLocalModel().split(currentSamplesVector[iSample]));

				/**
				 * Create the child nodes of the current node and call recursively to
				 * getNewTree()
				 */
				currentData = null;
				currentSamplesVector = null;
				currentTree.createSonsVector(currentTree.getLocalModel().numSubsets());
				for (int iSample = 0; iSample < numberSamples; iSample++)
					((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[iSample])
							.createSonsVector(currentTree.getLocalModel().numSubsets());

				//////////////////
				C45ModelSelectionExtended baseModelToForceDecision = 
						currentTree.m_sampleTreeVector[0].getBaseModelToForceDecision();
				for (int iSon = 0; iSon < currentTree.getSons().length; iSon++) {
					/** Vector storing the subsamples related to the iSon-th son */
					Instances[] localSamplesVector = new Instances[numberSamples];
					for (int iSample = 0; iSample < numberSamples; iSample++)
						localSamplesVector[iSample] = ((Instances[]) localInstancesVector.get(iSample))[iSon];

					// getNewTree
					C45ItPartiallyConsolidatedPruneableClassifierTree newTree = new C45ItPartiallyConsolidatedPruneableClassifierTree(
							currentTree.getToSelectModel(), baseModelToForceDecision, m_pruneTheTree, m_CF,
							m_subtreeRaising, m_cleanup, m_collapseTheTree, localSamplesVector.length,
							m_numberConsoNodesHowToSet,
							m_priorityCriteria, m_heuristicSearchAlgorithm, m_pruneTheConsolidatedTree, m_collapseTheCTree,
							m_pruneBaseTreesWithoutPreservingConsolidatedStructure);

					/** Set the recent created base trees like the sons of the given parent node */
					for (int iSample = 0; iSample < numberSamples; iSample++)
						((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[iSample]).setIthSon(iSon,
								newTree.m_sampleTreeVector[iSample]);

					ClassifierSplitModel sonModel;
					if ((m_priorityCriteria == J48PartiallyConsolidated.PriorCrit_GainratioWholeData) ||
							(m_priorityCriteria == J48PartiallyConsolidated.PriorCrit_GainratioWholeData_Size))
						sonModel = newTree.getToSelectModel().
								selectModel(localInstances[iSon]);
					else
						sonModel = ((C45ConsolidatedModelSelection)newTree.getToSelectModel()).
								selectModel(localInstances[iSon], localSamplesVector);

					if (sonModel.numSubsets() > 1) {
						orderValue = ((C45Split) sonModel).gainRatio();
						if ((m_priorityCriteria == J48PartiallyConsolidated.PriorCrit_GainratioWholeData_Size) ||
								(m_priorityCriteria == J48PartiallyConsolidated.PriorCrit_GainratioSetSamples_Size)) {
							double size = currentTree.getLocalModel().distribution().perBag(iSon);
							orderValue = orderValue * size;
						}
					}
					else
						orderValue = (double) Double.MIN_VALUE;
					
					Object[] son = new Object[]{localInstances[iSon], localSamplesVector, newTree, orderValue};
					if (m_heuristicSearchAlgorithm == J48PartiallyConsolidated.SearchAlg_HillClimbing)
						addSonOrderedByValue(listSons, son);
					else // SearchAlg_BestFirst
						addSonOrderedByValue(list, son);

					currentTree.setIthSon(iSon, newTree);

					localInstances[iSon] = null;
					localSamplesVector = null;
				}

				if (m_heuristicSearchAlgorithm == J48PartiallyConsolidated.SearchAlg_HillClimbing) {
					listSons.addAll(list);
					list = listSons;
				}

				localInstances = null;
				localInstancesVector.clear();
				listSons = null;
				internalNodes++;

			} else {
				currentTree.setIsLeaf(true);
				for (int iSample = 0; iSample < numberSamples; iSample++)
					currentTree.m_sampleTreeVector[iSample].setIsLeaf(true);

				if (Utils.eq(currentTree.getLocalModel().distribution().total(), 0)) {
					currentTree.setIsEmpty(true);
					for (int iSample = 0; iSample < numberSamples; iSample++)
						currentTree.m_sampleTreeVector[iSample].setIsEmpty(true);
				}
				currentData = null;
				currentSamplesVector = null;
			}
			index++;
		}
	}
}
