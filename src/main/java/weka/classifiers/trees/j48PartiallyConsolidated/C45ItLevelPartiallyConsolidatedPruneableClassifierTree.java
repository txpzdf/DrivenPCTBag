package weka.classifiers.trees.j48PartiallyConsolidated;

import java.util.ArrayList;

import weka.classifiers.trees.J48PartiallyConsolidated;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a 'partial' consolidated tree structure,
 * based on a consolidation percentage or on a specific value,
 * that indicates the number of levels of the tree to be developed. 
 * The tree is processed iteratively (instead of recursively like 
 * the original version).
 * (m_priorityCriteria == PriorCrit_Levelbylevel)
 * 
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @author Josué Cabezas Regoyo
 * @version $Revision: 1.0 $
 */
public class C45ItLevelPartiallyConsolidatedPruneableClassifierTree
		extends C45ItPartiallyConsolidatedPruneableClassifierTree {

	private static final long serialVersionUID = 7265782633965535703L;

	public C45ItLevelPartiallyConsolidatedPruneableClassifierTree(ModelSelection toSelectLocModel,
			C45ModelSelectionExtended baseModelToForceDecision, boolean pruneTree, float cf, boolean raiseTree,
			boolean cleanup, boolean collapseTree, int numberSamples, int numberConsoNodesHowToSet,
			int priorityCriteria, int heuristicSearchAlgorithm, boolean pruneCT, boolean collapseCT,
			boolean notPreservingStructure) throws Exception {
		super(toSelectLocModel, baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree, numberSamples,
				numberConsoNodesHowToSet, priorityCriteria, heuristicSearchAlgorithm, pruneCT, collapseCT,
				notPreservingStructure);
	}

	/**
	 * Determines the number of levels of the partial tree to be developed based 
	 * on a percentage value with respect to the number of levels of the whole 
	 * consolidated tree (which must first be constructed) or based on a specific 
	 * value (passed as a parameter).
	 * 
	 * @param data                 the data for pruning the consolidated tree
	 * @param samplesVector        the vector of samples for building the
	 *                             consolidated tree
	 * @param consolidationPercent the value of consolidation percent
	 * @throws Exception if something goes wrong
	 */
	public void setNumberNodesToBeConsolidated(Instances data, Instances[] samplesVector, float consolidationPercent) throws Exception {
		if (m_numberConsoNodesHowToSet == J48PartiallyConsolidated.NumberConsoNodes_Percentage) {
			long trainTimeStart = 0, trainTimeElapsed = 0;
			
			trainTimeStart = System.currentTimeMillis();
			super.buildTree(data, samplesVector, m_subtreeRaising || !m_cleanup); // build the tree without restrictions
							
			if (m_collapseTheCTree) {
				collapse();
			}
			if (m_pruneTheConsolidatedTree) {
				prune();
			}
			trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
			System.out.println("Time taken to build the whole consolidated tree: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
			m_elapsedTimeTrainingWholeCT = trainTimeElapsed / (double)1000.0;

			// m_priorityCriteria == J48PartiallyConsolidated.PriorCrit_Levelbylevel

			// Number of levels of the consolidated tree
			int treeLevels = numLevels();

			// Number of levels of the consolidated tree to leave as consolidated based on
			// given consolidationPercent
			int numberLevelsConso = (int) (((treeLevels * consolidationPercent) / 100) + 0.5);
			m_maximumCriteria = numberLevelsConso;
			setNumInternalNodesConso(numberLevelsConso);
			System.out.println(
					"Number of levels to leave as consolidated: " + numberLevelsConso + " of " + treeLevels);
		} else { // m_numberConsoNodesHowToSet == J48PartiallyConsolidated.NumberConsoNodes_Value
			m_maximumCriteria = (int) consolidationPercent;
			System.out.println("Number of levels to leave as consolidated: " + m_maximumCriteria);
			m_elapsedTimeTrainingWholeCT = (double)0.0;
		}
	}
	
	/**
	 * Builds the partial consolidated tree structure, in this case
	 * iteratively (instead of recursively as in the original method, buildTree()).
	 * (m_priorityCriteria == PriorCrit_Levelbylevel)
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
		int currentLevel;
		int index = 0;

		/** Initialize the consolidated tree */
		initiliazeTree(data, keepData);
		/** Initialize the base trees */
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample].initiliazeTree(samplesVector[iSample], keepData);

		/** List of nodes to be processed */
		ArrayList<Object[]> list = new ArrayList<>();

		// add(Data, samplesVector, tree, orderValue, currentLevel)
		list.add(new Object[] { data, samplesVector, this, null, 0 }); // The parent node is considered level 0

		while (list.size() > 0) {

			Object[] current = list.get(0);
			list.set(0, null); // Null to free up memory
			list.remove(0);

			currentData = (Instances)current[0];
			currentSamplesVector = (Instances[])current[1];
			currentTree = (C45ItLevelPartiallyConsolidatedPruneableClassifierTree)current[2];
			currentLevel = (int)current[4];

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

			if ((currentTree.getLocalModel().numSubsets() > 1) && (currentLevel < m_maximumCriteria)) {

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
					C45ItPartiallyConsolidatedPruneableClassifierTree newTree = new C45ItLevelPartiallyConsolidatedPruneableClassifierTree(
							currentTree.getToSelectModel(), baseModelToForceDecision, m_pruneTheTree, m_CF,
							m_subtreeRaising, m_cleanup, m_collapseTheTree, localSamplesVector.length,
							m_numberConsoNodesHowToSet,
							m_priorityCriteria, m_heuristicSearchAlgorithm, m_pruneTheConsolidatedTree, m_collapseTheCTree,
							m_pruneBaseTreesWithoutPreservingConsolidatedStructure);

					/** Set the recent created base trees like the sons of the given parent node */
					for (int iSample = 0; iSample < numberSamples; iSample++)
						((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[iSample]).setIthSon(iSon,
								newTree.m_sampleTreeVector[iSample]);

					listSons.add(new Object[]{localInstances[iSon], localSamplesVector, newTree, null, currentLevel+1});

					currentTree.setIthSon(iSon, newTree);

					localInstances[iSon] = null;
					localSamplesVector = null;
				}
				list.addAll(listSons);

				localInstances = null;
				localInstancesVector.clear();
				listSons = null;
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
