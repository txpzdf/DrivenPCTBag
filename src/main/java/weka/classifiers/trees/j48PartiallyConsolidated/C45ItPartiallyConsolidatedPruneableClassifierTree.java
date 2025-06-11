/**
 *
 */
package weka.classifiers.trees.j48PartiallyConsolidated;


import java.util.ArrayList;

import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.J48PartiallyConsolidated;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a 'partial' consolidated tree structure
 * (based on a consolidation percentage or on a specific value) but
 * the tree is processed iteratively (instead of recursively like 
 * the original version) and allows to decide which will be the 
 * next node to be developed according to a priority criterion.
 *
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @author Josué Cabezas Regoyo
 * @version $Revision: 1.0 $
 */
public class C45ItPartiallyConsolidatedPruneableClassifierTree extends C45PartiallyConsolidatedPruneableClassifierTree {

	/** for serialization **/
	private static final long serialVersionUID = 6410655550027990502L;

	/** Indicates the order in which the node was treated */
	protected int m_order;

	/**
	 * Builds the tree up to a maximum of depth levels. Set m_maximumLevel to 0 for
	 * default.
	 */
	protected int m_maximumCriteria;

	/** Selected way to set the number of nodes to be developed from the partial consolidated tree;
	 * based on a percentage value or by using a concrete value.
	 */
	protected int m_numberConsoNodesHowToSet;

	/** Indicates the criteria that should be used to build the tree */
	protected int m_priorityCriteria;
	
	/** Indicates the heuristic search algorithm that should be used to build the tree */
	protected int m_heuristicSearchAlgorithm;

	/** True if the partial Consolidated tree(CT) is to be pruned. */
	protected boolean m_pruneTheConsolidatedTree = false;
	
	/** True if the partial Consolidated tree(CT) is to be collapsed. */
	protected boolean m_collapseTheCTree = false;

	/** Time taken to build the whole Consolidated Tree (CT), including pruning/collapsing, if required. */
	protected double m_elapsedTimeTrainingWholeCT = (double)Double.NaN;

	/** Time taken to build the partial Consolidated Tree (CT), including pruning/collapsing, if required. */
	protected double m_elapsedTimeTrainingPartialCT = (double)Double.NaN;

	/** Time taken to build all base trees that compose the final multiple classifier (Bagging), including pruning/collapsing, if required. */
	protected double m_elapsedTimeTrainingAssocBagging = (double)Double.NaN;

	/** Whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree. */
	protected boolean m_pruneBaseTreesWithoutPreservingConsolidatedStructure;

	/**  Number of base trees preserving the split decision of the current consolidated node. */
	protected int m_numberBaseTreesWithThisSplitDecision = 0;
	
	/** Average percentage of base trees preserving structure throughout the tree. */
	protected double m_avgPercBaseTreesPreservingStructure = (double)Double.NaN;
	
	/** Minimum percentage of base trees preserving structure throughout the tree. */
	protected double m_minPercBaseTreesPreservingStructure = (double)Double.NaN;
	
	/** Maximum percentage of base trees preserving structure throughout the tree. */
	protected double m_maxPercBaseTreesPreservingStructure = (double)Double.NaN;
	
	/** Median percentage of base trees preserving structure throughout the tree. */
	protected double m_mdnPercBaseTreesPreservingStructure = (double)Double.NaN;
	
	/** Standard Deviation percentage of base trees preserving structure throughout the tree. */
	protected double m_devPercBaseTreesPreservingStructure = (double)Double.NaN;
	
	/**
	 * Constructor for pruneable consolidated tree structure. Calls the superclass
	 * constructor.
	 *
	 * @param toSelectLocModel      selection method for local splitting model
	 * @param pruneTree             true if the tree is to be pruned
	 * @param cf                    the confidence factor for pruning
	 * @param raiseTree             true if subtree raising has to be performed
	 * @param cleanup               true if cleanup has to be done
	 * @param collapseTree          true if collapse has to be done
	 * @param numberSamples         Number of Samples
	 * @param numberConsoNodesHowToSet How to set the number of consolidated nodes
	 * @param ITPCTmaximumCriteria  maximum number of nodes or levels
	 * @param priorityCriteria criteria to build the tree
	 * @param heuristicSearchAlgorithm search algorithm to build the tree
	 * @param pruneCT true if the CT tree is to be pruned
	 * @param collapseCT true if the CT tree is to be collapsed
	 * @param notPreservingStructure false if the base trees do not preserve the structure of the partial tree after pruning.
	 * @throws Exception if something goes wrong
	 */
	public C45ItPartiallyConsolidatedPruneableClassifierTree(
			ModelSelection toSelectLocModel, C45ModelSelectionExtended baseModelToForceDecision,
			boolean pruneTree, float cf,
			boolean raiseTree, boolean cleanup, 
			boolean collapseTree, int numberSamples,
			int numberConsoNodesHowToSet,
			int priorityCriteria, int heuristicSearchAlgorithm, 
			boolean pruneCT, boolean collapseCT,
			boolean notPreservingStructure) throws Exception {
		super(toSelectLocModel, baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree,
				numberSamples, notPreservingStructure);

		m_numberConsoNodesHowToSet = numberConsoNodesHowToSet;
		m_priorityCriteria = priorityCriteria;
		m_heuristicSearchAlgorithm = heuristicSearchAlgorithm;
		m_pruneTheConsolidatedTree = pruneCT;
		m_collapseTheCTree = collapseCT;
		m_pruneBaseTreesWithoutPreservingConsolidatedStructure=notPreservingStructure;
		m_numberBaseTreesWithThisSplitDecision=0;
	}

	/**
	 * Method for building a pruneable classifier partial consolidated tree;
	 * but iteratively; and directly, if a specific number of nodes to be 
	 * developed is indicated (instead of the consolidation percentage),
	 * being driven on the basis of a priority criterion. 
	 *
	 * @param data                 the data for pruning the consolidated tree
	 * @param samplesVector        the vector of samples for building the
	 *                             consolidated tree
	 * @param consolidationPercent the value of consolidation percent
	 * @throws Exception if something goes wrong
	 */
	public void buildClassifier(Instances data, Instances[] samplesVector, float consolidationPercent) throws Exception {
		long trainTimeStart = 0, trainTimeElapsed = 0;

		setNumberNodesToBeConsolidated(data, samplesVector, consolidationPercent);
		// buildTree
		trainTimeStart = System.currentTimeMillis();
		buildPartialTreeItera(data, samplesVector, m_subtreeRaising || !m_cleanup);
		if (m_collapseTheCTree) {
			collapse();
		}
		if (m_pruneTheConsolidatedTree) {
			prune();
		}
		trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
		System.out.println("Time taken to build the partial consolidated tree: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
		m_elapsedTimeTrainingPartialCT = trainTimeElapsed / (double)1000.0;

		trainTimeStart = System.currentTimeMillis();
		applyBagging();
		trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
		System.out.println("Time taken to build the associated Bagging: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
		m_elapsedTimeTrainingAssocBagging = trainTimeElapsed / (double)1000.0;

		if (m_cleanup)
			cleanup(new Instances(data, 0));
		if(!m_isLeaf)
			computeNumberBaseTreesPreservingPartialCTStructure();
	}

	/**
	 * Determines the number of nodes (or levels) of the partial tree to be developed 
	 * based on a percentage value with respect to the number of inner nodes (or levels) 
	 * of the whole consolidated tree (which must first be constructed) or based on a 
	 * specific value (passed as a parameter).
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

			// Number of internal nodes of the consolidated tree
			int innerNodes = numNodes() - numLeaves();

			// Number of nodes of the consolidated tree to leave as consolidated based on
			// given consolidationPercent
			int numberNodesConso = (int) (((innerNodes * consolidationPercent) / 100) + 0.5);
			m_maximumCriteria = numberNodesConso;
			setNumInternalNodesConso(numberNodesConso);
			System.out.println(
					"Number of nodes to leave as consolidated: " + numberNodesConso + " of " + innerNodes);
		} else { // m_numberConsoNodesHowToSet == J48PartiallyConsolidated.NumberConsoNodes_Value
			m_maximumCriteria = (int) consolidationPercent;
			System.out.println("Number of nodes or levels to leave as consolidated: " + m_maximumCriteria);
			m_elapsedTimeTrainingWholeCT = (double)0.0;
		}
	}

	/**
	 * Builds the partial consolidated tree structure, in this case
	 * iteratively (instead of recursively as in the original method, buildTree()).
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
		list.add(new Object[]{data, samplesVector, this, null}); // The parent node is considered level 0

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

					if ((m_priorityCriteria >= J48PartiallyConsolidated.PriorCrit_GainratioWholeData) &&
								(m_priorityCriteria <= J48PartiallyConsolidated.PriorCrit_GainratioSetSamples_Size)) // Added by gainratio,
																							// largest to smallest
					{
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
						if (m_heuristicSearchAlgorithm == J48PartiallyConsolidated.SearchAlg_BestFirst)
							addSonOrderedByValue(list, son);
						else
							addSonOrderedByValue(listSons, son);
					} else
						listSons.add(new Object[]{localInstances[iSon], localSamplesVector, newTree, null});

					currentTree.setIthSon(iSon, newTree);

					localInstances[iSon] = null;
					localSamplesVector = null;
				}

				if ((m_priorityCriteria == J48PartiallyConsolidated.PriorCrit_Preorder) ||
					(m_heuristicSearchAlgorithm == J48PartiallyConsolidated.SearchAlg_HillClimbing)) {
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

	public void addSonOrderedByValue(ArrayList<Object[]> list, Object[] son) {
		if (list.size() == 0) {
			list.add(son);
		} else {
			int i;
			double sonValue = (double) son[3];
			for (i = 0; i < list.size(); i++) {
				double parentValue = (double) list.get(i)[3];
				if (parentValue < sonValue) {
					list.add(i, son);
					break;
				}
			}
			if (i == list.size())
				list.add(son);
		}
	}

	/**
	 * Initializes the base tree to be build.
	 * @param data instances in the current node related to the corresponding base decision tree
	 * @param keepData  is training data to be kept?
	 */
	public void initiliazeTree(Instances data, boolean keepData) {
		if (keepData) {
			m_train = data;
		}
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
	}

	/**
	 * Setter for m_isLeaf member.
	 * @param isLeaf indicates if node is leaf
	 */
	public void setIsLeaf(boolean isLeaf) {
		m_isLeaf = isLeaf;
	}

	/**
	 * Setter for m_isEmpty member.
	 * @param isEmpty indicates if node is empty
	 */
	public void setIsEmpty(boolean isEmpty) {
		m_isEmpty = isEmpty;
	}

	/**
	 * Set m_localModel based on the consolidation proccess.
	 * @param data instances in the current node
	 * @param samplesVector the vector of samples used for consolidation
	 * @throws Exception if something goes wrong
	 */
	public void setLocalModel(Instances data, Instances[] samplesVector) throws Exception {
		m_localModel = ((C45ConsolidatedModelSelection)m_toSelectModel).selectModel(data, samplesVector);
	}

	/**
	 * Creates the vector to save the sons of the current node.
	 * @param numSons Number of sons
	 */
	public void createSonsVector(int numSons) {
		m_sons = new ClassifierTree [numSons];
	}
	
	/**
	 * Set given baseTree tree like the i-th son tree.
	 * @param iSon Index of the vector to save the given tree
	 * @param classifierTree the given to tree to save
	 */
	public void setIthSon(int iSon, ClassifierTree classifierTree) {
		m_sons[iSon] = classifierTree;
	}

	/**
	 * Getter for m_toSelectModel member.
	 * return the model selection method
	 */
	public ModelSelection getToSelectModel() {
		return m_toSelectModel;
	}

	/**
	 * Help method for printing tree structure.
	 * 
	 * @param depth the current depth
	 * @param text  for outputting the structure
	 * @throws Exception if something goes wrong
	 */
	public void dumpTree(int depth, StringBuffer text) throws Exception {

		int i, j;
		int numberSamples = m_sampleTreeVector.length;

		for (i = 0; i < m_sons.length; i++) {
			text.append("\n");
			;
			for (j = 0; j < depth; j++) {
				text.append("|   ");
			}
			text.append("[" + m_order + "]");
			if(m_pruneBaseTreesWithoutPreservingConsolidatedStructure)
				text.append("[Str: " + Utils.doubleToString(m_numberBaseTreesWithThisSplitDecision*(double)100/numberSamples,2) + "%]");
			text.append(m_localModel.leftSide(m_train));
			text.append(m_localModel.rightSide(i, m_train));
			if (m_sons[i].isLeaf()) {
				text.append(": ");
				text.append("[" + ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_sons[i]).m_order + "] ");
				text.append(m_localModel.dumpLabel(i, m_train));
			} else {
				m_sons[i].dumpTree(depth + 1, text);
			}
		}
	}

	/**
	 * @return the m_elapsedTimeTrainingWholeCT
	 */
	public double getElapsedTimeTrainingWholeCT() {
		return m_elapsedTimeTrainingWholeCT;
	}

	/**
	 * @return the m_elapsedTimeTrainingPartialCT
	 */
	public double getElapsedTimeTrainingPartialCT() {
		return m_elapsedTimeTrainingPartialCT;
	}

	/**
	 * @return the m_elapsedTimeTrainingAssocBagging
	 */
	public double getElapsedTimeTrainingAssocBagging() {
		return m_elapsedTimeTrainingAssocBagging;
	}
	
	/**
	 * @return the m_avgPercBaseTreesPreservingStructure
	 */
	public double getAvgPercBaseTreesPreservingStructure() {
		return m_avgPercBaseTreesPreservingStructure;
	}

	/**
	 * @return the m_minPercBaseTreesPreservingStructure
	 */
	public double getMinPercBaseTreesPreservingStructure() {
		return m_minPercBaseTreesPreservingStructure;
	}

	/**
	 * @return the m_maxPercBaseTreesPreservingStructure
	 */
	public double getMaxPercBaseTreesPreservingStructure() {
		return m_maxPercBaseTreesPreservingStructure;
	}

	/**
	 * @return the m_mdnPercBaseTreesPreservingStructure
	 */
	public double getMdnPercBaseTreesPreservingStructure() {
		return m_mdnPercBaseTreesPreservingStructure;
	}

	/**
	 * @return the m_devPercBaseTreesPreservingStructure
	 */
	public double getDevPercBaseTreesPreservingStructure() {
		return m_devPercBaseTreesPreservingStructure;
	}

	/**
	 * Computes the number of base trees that preserve the same structure as
	 * the partial consolidated tree.
	 */
	public void computeNumberBaseTreesPreservingPartialCTStructure() {
		if(m_pruneBaseTreesWithoutPreservingConsolidatedStructure) {
			/** Number of Samples. */
			int numberSamples = m_sampleTreeVector.length;
			m_numberBaseTreesWithThisSplitDecision = 0;
			for (int iSample = 0; iSample < numberSamples; iSample++)
				computeWhetherBaseTreePreservesStructure((C45PruneableClassifierTreeExtended)(m_sampleTreeVector[iSample]));
			ArrayList<Double> auxvPercBaseTrees = new ArrayList<>();
			getAllPercBaseTreesPreservingStructure(auxvPercBaseTrees);
			//m_avgPercBaseTreesPreservingStructure = auxvPercBaseTrees.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
			double[] vPercBaseTrees = auxvPercBaseTrees.stream().mapToDouble(Double::doubleValue).toArray();
			m_avgPercBaseTreesPreservingStructure = Utils.mean(vPercBaseTrees);
			m_minPercBaseTreesPreservingStructure = vPercBaseTrees[Utils.minIndex(vPercBaseTrees)];
	        m_maxPercBaseTreesPreservingStructure = vPercBaseTrees[Utils.maxIndex(vPercBaseTrees)];
	        m_mdnPercBaseTreesPreservingStructure = Utils.kthSmallestValue(vPercBaseTrees, vPercBaseTrees.length / 2);
	        m_devPercBaseTreesPreservingStructure = Math.sqrt(Utils.variance(vPercBaseTrees));
		} else {
			m_avgPercBaseTreesPreservingStructure = (double)100.0;
			m_minPercBaseTreesPreservingStructure = (double)100.0;
	        m_maxPercBaseTreesPreservingStructure = (double)100.0;
	        m_mdnPercBaseTreesPreservingStructure = (double)100.0;
	        m_devPercBaseTreesPreservingStructure = (double)0.0;
		}
	}
	
	public void computeWhetherBaseTreePreservesStructure(ClassifierTree baseTree) {
		if (m_isLeaf)
			return;
		if (baseTree.isLeaf())
			return;
		int attPCT=((C45Split)m_localModel).attIndex();
		double pointPCT=((C45Split)m_localModel).splitPoint();
		if((attPCT==((C45Split)baseTree.getLocalModel()).attIndex())&&
			(Utils.eq(pointPCT,((C45Split)baseTree.getLocalModel()).splitPoint()))) {
			m_numberBaseTreesWithThisSplitDecision+=1;
			ClassifierTree[] vsons=baseTree.getSons();
			for (int i=0;i<m_sons.length;i++)
				((C45ItPartiallyConsolidatedPruneableClassifierTree)son(i)).computeWhetherBaseTreePreservesStructure(vsons[i]);
		} else {
			// If the node has children and the split does not match, subtree raising has occurred in the pruning.
			int indexOfLargestBranch = localModel().distribution().maxBag();
			((C45ItPartiallyConsolidatedPruneableClassifierTree)son(indexOfLargestBranch)).computeWhetherBaseTreePreservesStructure(baseTree);
		}
	}
	
	public void getAllPercBaseTreesPreservingStructure(ArrayList<Double> list) {
		/** Number of Samples. */
		int numberSamples = m_sampleTreeVector.length;
		if(!m_isLeaf) {
			list.add(m_numberBaseTreesWithThisSplitDecision*(double)100/numberSamples);
			for (int i=0;i<m_sons.length;i++)
				((C45ItPartiallyConsolidatedPruneableClassifierTree)son(i)).getAllPercBaseTreesPreservingStructure(list);
		}
	}
}
