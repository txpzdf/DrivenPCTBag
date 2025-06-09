/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    PartiallyConsolidatedTreeBagging.java
 *    Copyright (C) 2021 ALDAPA Team (http://www.aldapa.eus)
 *    Faculty of Informatics, Donostia, 20018
 *    University of the Basque Country (UPV/EHU), Basque Country
 *
 */

package weka.classifiers.trees;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ItPartiallyConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ModelSelectionExtended;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PartiallyConsolidatedPruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Summarizable;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
<!-- globalinfo-start -->
 * Class for generating a Partially Consolidated Tree-Bagging (PCTBagging) multiple classifier.<br/>
 * Allows building a classifier between a single consolidated tree (100%), based on J48Consolidated,
 * and a bagging (0%), according to the given consolidation percent. First, a partially consolidated
 * tree is built based on a set of samples, and then, a standard J48 decision tree is developed from
 * the leaves of the tree related to each sample, as Bagging does.<p/>
 * For more information, see:<p/>
 * Igor Ibarguren and Jes&uacute;s M. P&eacute;rez and Javier Muguerza and Olatz Arbelaitz and Ainhoa Yera.  
 * "PCTBagging: From Inner Ensembles to Ensembles. A trade-off between Discriminating Capacity and Interpretability". Information Sciences (2022), Vol. 583, pp 219-238.
 * <a href="https://doi.org/10.1016/j.ins.2021.11.010" target="_blank">doi:10.1016/j.ins.2021.11.010</a>
 * <p/>
 * Also see:<br/>
 * <br/>
 * Igor Ibarguren and Jes&uacute;s M. P&eacute;rez and Javier Muguerza and Ibai Gurrutxaga and Olatz Arbelaitz.  
 * "Coverage-based resampling: Building robust consolidated decision trees". Knowledge Based Systems (2015), Vol. 79, pp 51-67.
 * <a href="http://dx.doi.org/10.1016/j.knosys.2014.12.023" target="_blank">doi:10.1016/j.knosys.2014.12.023</a>
 * <p/>
<!-- globalinfo-end -->
 *
<!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Ibarguren2022,
 *    title = "PCTBagging: From Inner Ensembles to Ensembles. A trade-off between Discriminating Capacity and Interpretability",
 *    journal = "Information Sciences",
 *    volume = "583",
 *    pages = "219 - 238",
 *    year = "2022",
 *    doi = "10.1016/j.ins.2021.11.010",
 *    author = "Igor Ibarguren and Jes\'us M. P\'erez and Javier Muguerza and Olatz Arbelaitz and Ainhoa Yera"
 * }
 * </pre>
 * <p/>
 * <pre>
 * &#64;article{Ibarguren2015,
 *    title = "Coverage-based resampling: Building robust consolidated decision trees",
 *    journal = "Knowledge Based Systems",
 *    volume = "79",
 *    pages = "51 - 67",
 *    year = "2015",
 *    doi = "10.1016/j.knosys.2014.12.023",
 *    author = "Igor Ibarguren and Jes\'us M. P\'erez and Javier Muguerza and Ibai Gurrutxaga and Olatz Arbelaitz"
 * }
 * </pre>
 * <p/>
<!-- technical-bibtex-end -->
 * *************************************************************************************<br/>
 * Attention! Removed 'final' modifier of 'j48.distributionForInstance(...)'
 *  in order to override this function here.
 * <ul>
 *    <li>public double [] distributionForInstance(Instance instance)</li>
 * </ul>
 * *************************************************************************************<br/>
<!-- options-start -->
 * Valid options are: <p/>
 * 
 * J48 options <br/>
 * ==========
 *
 * <pre> -U
 *  Use unpruned tree.</pre>
 * 
 * <pre> -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)</pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)</pre>
 *  
 * <pre> -S
 *  Don't perform subtree raising.</pre>
 * 
 * <pre> -L
 *  Do not clean up after the tree has been built.</pre>
 * 
 * <pre> -A
 *  Laplace smoothing for predicted probabilities.</pre>
 * 
 * <pre> -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 * 
 * Options to set the Resampling Method (RM) for the generation of samples
 *  to use in the consolidation process <br/>
 * =============================================================================================== 
 * <pre> -RM-C
 *  Determines that the way to set the number of samples to be generated will be based on
 *  a coverage value as a percent. In the case this option is not set, the number of samples
 *  will be determined using a fixed value. 
 *  (Set by default)</pre>
 * 
 * <pre> -RM-N &lt;number of samples&gt;
 *  Number of samples to be generated for use in the construction of the consolidated tree.
 *  It can be set as a fixed value or based on a coverage value as a percent, when -RM-C option
 *  is used, which guarantees the number of samples necessary to cover adequately the examples 
 *  of the original sample
 *  (default 5 for a fixed value or 99% for the case based on a coverage value)</pre>
 * 
 * <pre> -RM-R
 *  Determines whether or not replacement is used when generating the samples.
 *  (default false)</pre>
 * 
 * <pre> -RM-B &lt;Size of each sample(&#37;)&gt;
 *  Size of each sample(bag), as a percentage of the training set size.
 *  Combined with the option &lt;distribution minority class&gt; accepts:
 *  * -1 (sizeOfMinClass): The size of the minority class  
 *  * -2 (Max): Maximum size taking into account &lt;distribution minority class&gt;
 *  *           and using no replacement
 *  (default -2)</pre>
 *  
 * <pre> -RM-D &lt;distribution minority class&gt;
 *  Determines the new value of the distribution of the minority class, if we want to change it.
 *  It can be one of the following values:
 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
 *    (this option can only be used with binary problems (two classes datasets))
 *  * -1 (free): Works with the instances without taking into account their class  
 *  * -2 (stratified): Maintains the original class distribution in the new samples
 *  (default 50.0)</pre>
 * 
 * Options to leave partially consolidated the built consolidated tree (PCTB)
 * ============================================================================ 
 * <pre> -PCTB-C consolidation percent <br>
 * Determines the number of inner nodes to leave as consolidated as a percent 
 * according to the inner nodes of the whole consolidated tree.  
 * (Default: 20.0)<pre>
 * 
 * <pre> -PCTB-V mode <br>
 * Determines how many base trees will be shown:
 *  None, only the first ten (if they exist) or all.  
 * (Default: only the first ten)<pre>
 * 
 * <pre> -PCTB-P<br>
 * Determines whether to show the explanation aggregated measures of the all decision trees
 *  that compose the final classifier (MCS).  
 * (Default: false)<pre>
 * 
 * <pre> -PCTB-WP<br>
 * Determines whether to prune the base trees without preserving the structure of
 * the partially consolidated tree. 
 * (Default: false)<pre>
 * 
<!-- options-end -->
 *
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @author Ander Otsoa de Alda Alzaga (ander.otsoadealda@gmail.com)
 *  (based on J48Consolidated.java written by Jesús M. Pérez et al) 
 *  (based on J48.java written by Eibe Frank)
 * @version $Revision: 0.3 $
 */
public class J48PartiallyConsolidated
	extends J48Consolidated
	implements OptionHandler, Drawable, Matchable, Sourcable,
				WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer,
				TechnicalInformationHandler {

	/** for serialization */
	private static final long serialVersionUID = 1279412716088899857L;

	/** Ways to set the number of consolidated nodes to be developed */
	public static final int NumberConsoNodes_Value = 1;
	public static final int NumberConsoNodes_Percentage = 2;

	/** Strings related to the ways to set the number of consolidated nodes to be developed */
	public static final Tag[] TAGS_WAYS_TO_SET_NUMBER_CONSOLIDATED_NODES = {
			new Tag(NumberConsoNodes_Value, "using a specific value"),
			new Tag(NumberConsoNodes_Percentage, "based on a pertentage value (%)"),
	};

	/** Ways to set the priority criteria option */
	public static final int PriorCrit_Original = 0;
	public static final int PriorCrit_Levelbylevel = 1;
	public static final int PriorCrit_Preorder = 2;
	public static final int PriorCrit_Size = 3;
	public static final int PriorCrit_GainratioWholeData = 4;
	public static final int PriorCrit_GainratioSetSamples = 5;
	public static final int PriorCrit_GainratioWholeData_Size = 6;
	public static final int PriorCrit_GainratioSetSamples_Size = 7;

	
	/** Strings related to the ways to set the priority criteria option */
	public static final Tag[] TAGS_WAYS_TO_SET_PRIORITY_CRITERIA = {
			new Tag(PriorCrit_Original, "Original (recursive)"),
			new Tag(PriorCrit_Levelbylevel, "Level by level"),
			new Tag(PriorCrit_Preorder, "Node by node - Pre-order"),
			new Tag(PriorCrit_Size, "Node by node - Size"),
			new Tag(PriorCrit_GainratioWholeData, "Node by node - Gain ratio (Whole data)"),
			new Tag(PriorCrit_GainratioSetSamples, "Node by node - Gain ratio (Set of samples)"),
			new Tag(PriorCrit_GainratioWholeData_Size, "Node by node - Gain ratio (Whole data) * Size"),
			new Tag(PriorCrit_GainratioSetSamples_Size, "Node by node - Gain ratio (Set of samples) * Size")
	};
	
	/** Ways to set the heuristic search algorithm */
	public static final int SearchAlg_BestFirst = 0;
	public static final int SearchAlg_HillClimbing = 1;
	
	/** Strings related to the ways to set the heuristic search algorithm */
	public static final Tag[] TAGS_WAYS_TO_SET_SEARCH_ALGORITHM = {
			new Tag(SearchAlg_BestFirst, "Best-first (Original)"),
			new Tag(SearchAlg_HillClimbing, "Hill climbing (In-depth)")
	};
	
	/** Options to visualize the set of base trees */
	public static final int Visualize_None = 1;
	public static final int Visualize_FirstOnes = 2;
	public static final int Visualize_All = 3;

	/** Strings related to the options to visualize the set of base trees */
	public static final Tag[] TAGS_VISUALIZE_BASE_TREES = {
		new Tag(Visualize_None, "None"),
		new Tag(Visualize_FirstOnes, "Only the first ten"),
		new Tag(Visualize_All, "All")
	};
	
	/** Options related to PCTBagging algorithm
	 *  (Prefix PCTB added to the option names in order to appear together in the graphical interface)
	 ********************************************************************************/

	/** The consolidation percent (if PCTBnumberConsoNodesHowToSet = "based on a pertentage value (%)").
	 * It indicates the number of inner nodes to leave as consolidated as a percent 
	 * according to the inner nodes of the whole consolidated tree. It accepts values between 0.0 and 100.0.
	 * if PCTBnumberConsoNodesHowToSet = "using a specific value", it directly indicates the number of 
	 * internal nodes to be developed in the construction of the consolidated tree.
	 */
	protected float m_PCTBconsolidationPercent = (float)20.0;

	/** Selected way to set the number of nodes to be developed from the partial consolidated tree;
	 * based on a percentage value (e.g. 20%) with regard to the number of nodes in the whole 
	 * consolidated tree (this implies to build the whole consolidated tree first) or
	 * by using a concrete value (e.g. 5 nodes).
	 */
	private int m_PCTBconsolidationPercentHowToSet = NumberConsoNodes_Percentage;
	
	/** Indicates the criteria used when choosing the next node to be developed in the construction 
	 * of the partial consolidated tree: Size, Pre-order, Gain ratio...
	 * The default value is the size of the nodes (number of instances they contain). 
	 */
	private int m_PCTBpriorityCriteria = PriorCrit_Size;
	
	/** Indicates the heuristic search algorithm used to determine the order in which the partial 
	 * consolidated tree will be developed based on the priority criteria.
	 * If the priority criterion is based on a quantitative value (such as size or gain ratio),
	 * the best-first search algorithm will choose the best of the remaining nodes to be developed.
	 * However, if the algorithm is Hill climbing, the next node to be developed will always be
	 * one of the child nodes of the last developed node, i.e. it will give priority to developing
	 * the tree in depth.
	 */
	private int m_PCTBheuristicSearchAlgorithm = SearchAlg_HillClimbing;
	
	/** Unpruned partial Consolidated Tree (CT)?
	 * Thus the original ‘unpruned’ parameter (from J48) takes effect on the base trees of the Bagging
	 * after the construction of the partial consolidated tree.
	 * Thus the user can decide to prune (or not) the partial consolidated tree (with this parameter)
	 * and, independently, decide to prune (or not) the trees associated with the Bagging after the 
	 * construction of the partial consolidated tree (with the original parameter).
	 */
	private boolean m_PCTBunprunedCT = true;
	
	/** Collapse partial Consolidated Tree (CT)?
	 * Thus the original ‘collapseTree’ parameter (from J48) takes effect on the base trees of the Bagging
	 * after the construction of the partial consolidated tree.
	 * Thus the user can decide to collapse (or not) the partial consolidated tree (with this parameter)
	 * and, independently, decide to collapse (or not) the trees associated with the Bagging after the 
	 * construction of the partial consolidated tree (with the original parameter).
	 */
	private boolean m_PCTBcollapseCT = false;

	/** Whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree.
	 * In the original PCTBagging, the pruning of the Bagging base trees (if active) is 
	 * guaranteed never to exceed the structure of the partial consolidated tree. 
	 * This parameter, if true, allows to freely prune each of these trees.
	 */
	protected boolean m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure = false;

	/** Visualize the base trees: None, only the first ten (if they exist) or all */
	protected int m_PCTBvisualizeBaseTrees = Visualize_FirstOnes;

	/** Whether to show the explanation aggregated measures of the all decision trees
	 *  that compose the final classifier (MCS). */
	protected boolean m_PCTBprintExplanationMeasuresBaseTrees = false;

	/** Array for storing the generated base classifiers.
	 * (based on Bagging.java written by Eibe Frank eta al)
	 * */
	protected ClassifierTree[] m_Classifiers;

	/**
	 * Returns a string describing the classifier
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Class for generating a Partially Consolidated Tree-Bagging (PCTBagging) multiple classifier. "
				+ "Allows building a classifier between a single consolidated tree (100%), based on J48Consolidated, "
				+ "and a bagging (0%), according to the given consolidation percent.\n"
				+ "First, a partially consolidated tree is built based on a set of samples, and then, "
				+ "a standard J48 decision tree is developed from the leaves of the tree related to each sample, as Bagging does. "
				+ "For more information, see:\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Igor Ibarguren and Jesús M. Pérez and Javier Muguerza and Olatz Arbelaitz and Ainhoa Yera");
		result.setValue(Field.YEAR, "2022");
		result.setValue(Field.TITLE, "PCTBagging: From Inner Ensembles to Ensembles. A trade-off between Discriminating Capacity and Interpretability");
	    result.setValue(Field.JOURNAL, "Information Sciences");
	    result.setValue(Field.VOLUME, "583");
	    result.setValue(Field.PAGES, "219-238");
	    result.setValue(Field.URL, "https://doi.org/10.1016/j.ins.2021.11.010");
	    
	    TechnicalInformation additional = new TechnicalInformation(Type.ARTICLE);
	    additional.setValue(Field.AUTHOR, "Igor Ibarguren and Jesús M. Pérez and Javier Muguerza and Ibai Gurrutxaga and Olatz Arbelaitz");
	    additional.setValue(Field.YEAR, "2015");
	    additional.setValue(Field.TITLE, "Coverage-based resampling: Building robust consolidated decision trees");
	    additional.setValue(Field.JOURNAL, "Knowledge Based Systems");
	    additional.setValue(Field.VOLUME, "79");
	    additional.setValue(Field.PAGES, "51-67");
	    additional.setValue(Field.URL, "http://dx.doi.org/10.1016/j.knosys.2014.12.023");
	    result.add(additional);

		return result;
	}

	/**
	 * Generates the classifier.
	 * (based on buildClassifier() function of J48Consolidated class)
	 *
	 * @see weka.classifiers.trees.J48Consolidated#buildClassifier(weka.core.Instances)
	 */
	public void buildClassifier(Instances instances)
			throws Exception {

		// can classifier tree handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class before generate samples
		instances = new Instances(instances);
		instances.deleteWithMissingClass();
		
		//Generate as many samples as the number of samples with the given instances
		Instances[] samplesVector = generateSamples(instances);
	    //if (m_Debug)
	    //	printSamplesVector(samplesVector);

		/** Set the model selection method to determine the consolidated decisions */
	    ModelSelection modSelection;
		// TODO Implement the option binarySplits of J48
		modSelection = new C45ConsolidatedModelSelection(m_minNumObj, instances, 
				m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
		/** Set the model selection method to force the consolidated decision in each base tree*/
		C45ModelSelectionExtended baseModelToForceDecision = new C45ModelSelectionExtended(m_minNumObj, instances, 
				m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
		// TODO Implement the option reducedErrorPruning of J48
		C45PartiallyConsolidatedPruneableClassifierTree localClassifier;
		switch (m_PCTBpriorityCriteria) {
			case PriorCrit_Original:
				if (m_unpruned != m_PCTBunprunedCT)
					throw new Exception("If priority criteria is equal to 'Original', no distinction can be made between unpruned and unprunedCT; they must have the same value!");
				if (m_collapseTree != m_PCTBcollapseCT)
					throw new Exception("If priority criteria is equal to 'Original', no distinction can be made between collapseTree and collapseCT; they must have the same value!");
				if (m_PCTBheuristicSearchAlgorithm == SearchAlg_HillClimbing)
					throw new Exception("The ‘Hill climbing’ option for heuristicSearchAlgorithm was not implemented, when priority criteria is equal to Original!");
				if (m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure)
					throw new Exception("The pruneBaseTreesWithoutPreservingConsolidatedStructure option was not implemented, when priority criteria is equal to Original!");
				if (m_PCTBconsolidationPercentHowToSet == NumberConsoNodes_Value)
					throw new Exception("It is not possible to indicate the number of nodes to be consolidated as a specific value, when priority criteria is equal to Original!");
				localClassifier = new C45PartiallyConsolidatedPruneableClassifierTree(
						modSelection, baseModelToForceDecision,
						!m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup, m_collapseTree, samplesVector.length);
				break;
			default:
				localClassifier = new C45ItPartiallyConsolidatedPruneableClassifierTree(
						modSelection, baseModelToForceDecision,
						!m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup, m_collapseTree, samplesVector.length,
						m_PCTBconsolidationPercentHowToSet,
						m_PCTBpriorityCriteria, m_PCTBheuristicSearchAlgorithm, !m_PCTBunprunedCT, m_PCTBcollapseCT,
						m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure);
		}
		localClassifier.buildClassifier(instances, samplesVector, m_PCTBconsolidationPercent);

		m_root = localClassifier;
		m_Classifiers = localClassifier.getSampleTreeVector();
		// // We could get any base tree of the vector as root and use it in the graphical interface
		// // (for example, to visualize it)
		// //m_root = localClassifier.getSampleTreeIth(0);
		
		((C45ModelSelection) modSelection).cleanup();
		((C45ModelSelection) baseModelToForceDecision).cleanup();
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 * (based on Bagging.java)
	 * 
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if distribution can't be computed successfully
	 * 
	 * @see weka.classifiers.meta.Bagging#distributionForInstance(weka.core.Instance)
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {
		/** Number of Samples. */
		int numberSamples = m_Classifiers.length;
		double[] sums = new double[instance.numClasses()], newProbs;

		for (int i = 0; i < numberSamples; i++) {
			if (instance.classAttribute().isNumeric() == true) {
				sums[0] += m_Classifiers[i].classifyInstance(instance);
			} else {
				newProbs = m_Classifiers[i].distributionForInstance(instance, m_useLaplace);
				for (int j = 0; j < newProbs.length; j++)
					sums[j] += newProbs[j];
			}
		}
		if (instance.classAttribute().isNumeric() == true) {
			sums[0] /= numberSamples;
			return sums;
		} else if (Utils.eq(Utils.sum(sums), 0)) {
			return sums;
		} else {
			Utils.normalize(sums);
			return sums;
		}
	}

	/**
	 * Classifies an instance.
	 * (based on J48.java)
	 * @see weka.classifiers.trees.J48.classifyInstance(Instance)
	 * 
	 * @param instance the instance to classify
	 * @return the classification for the instance
	 * @throws Exception if instance can't be classified successfully
	 */
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		double[] distribution = new double[instance.numClasses()];

		distribution = distributionForInstance(instance);
		
		double maxProb = -1;
	    double currentProb;
	    int maxIndex = 0;
	    int j;

	    for (j = 0; j < instance.numClasses(); j++) {
	      currentProb = distribution[j];
	      if (Utils.gr(currentProb, maxProb)) {
	        maxIndex = j;
	        maxProb = currentProb;
	      }
	    }

	    return maxIndex;
	}


	/**
	 * Returns an enumeration describing the available options.
	 *
	 * Valid options are: <p>
	 * 
	 * J48 options
	 * ============= 
	 *
	 * -U <br>
	 * Use unpruned tree.<p>
	 *
	 * -C confidence <br>
	 * Set confidence threshold for pruning. (Default: 0.25) <p>
	 *
	 * -M number <br>
	 * Set minimum number of instances per leaf. (Default: 2) <p>
	 *
	 * -S <br>
	 * Don't perform subtree raising. <p>
	 *
	 * -L <br>
	 * Do not clean up after the tree has been built. <p>
	 *
	 * -A <br>
	 * If set, Laplace smoothing is used for predicted probabilites. <p>
	 *
	 * -Q seed <br>
	 * Seed for random data shuffling (Default: 1) <p>
	 * 
	 * Options to set the Resampling Method (RM) for the generation of samples
	 *  to use in the consolidation process
	 * ============================================================================ 
	 * -RM-C <br>
	 * Determines that the way to set the number of samples to be generated will be based on<br>
	 * a coverage value as a percent. In the case this option is not set, the number of samples<br>
	 * will be determined using a fixed value. <br>
	 * (Set by default)<p>
	 * 
	 * -RM-N &lt;number of samples&gt;<br>
	 * Number of samples to be generated for use in the construction of the consolidated tree.<br>
	 * It can be set as a fixed value or based on a coverage value as a percent, when -RM-C option<br>
	 * is used, which guarantees the number of samples necessary to cover adequately the examples<br> 
	 * of the original sample<br>
	 * (Default 5 for a fixed value or 99% for the case based on a coverage value)<p>
	 * 
	 * -RM-R <br>
	 * Determines whether or not replacement is used when generating the samples. <br>
	 * (Default: false)<p>
	 * 
	 * -RM-B percent <br>
	 * Size of each sample(bag), as a percentage of the training set size. <br>
	 * Combined with the option &lt;distribution minority class&gt; accepts: <br>
	 *  * -1 (sizeOfMinClass): The size of the minority class <br>
	 *  * -2 (maxSize): Maximum size taking into account &lt;distribution minority class&gt;
	 *              and using no replacement <br>
	 * (Default: -2(maxSize)) <p>
	 * 
	 * -RM-D distribution minority class <br>
	 * Determines the new value of the distribution of the minority class, if we want to change it. <br>
	 * It can be one of the following values: <br>
	 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples <br>
	 *    (If the dataset is multi-class, only the special value 50.0 will be accepted to balance the classes)
	 *  * -1 (free): Works with the instances without taking into account their class <br>
	 *  * -2 (stratified): Maintains the original class distribution in the new samples <br>
	 * (Default: -1(free)) <p>
	 * 
	 * Options to leave partially consolidated the built consolidated tree (PCTB)
	 * ============================================================================ 
	 * <pre>-PCTB-C consolidation percent <br>
	 * Determines the number of inner nodes to leave as consolidated as a percent 
	 * according to the inner nodes of the whole consolidated tree (If -PCTB-BP appears(by default).
	 * If -PCTB-BP does not appear, it directly determines the number of internal nodes to be developed
	 * as a specific value.    
	 * (Default: 20.0)<pre>
	 *
	 * <pre>-PCTB-BP<br>
	 * Set the way to set the number of inner nodes to be developed from the partial 
	 * consolidated tree based on a percentage value with regard to the number of nodes 
	 * in the whole consolidated tree (instead of using a concrete value).
	 * (Default: true)<pre>
	 * 
	 * <pre>-PCTB-PC criteria<br>
	 * Indicates the criteria used when choosing the next node to be developed in the construction
	 * of the partial consolidated tree:
	 *  · 0: Original (recursive)
	 *  · 1: Level by level
	 *  · 2: Node by node - Pre-order
	 *  · 3: Node by node - Size
	 *  · 4: Node by node - Gain ratio (Whole data)
	 *  · 5: Node by node - Gain ratio (Set of samples)
	 *  · 6: Node by node - Gain ratio (Whole data) x Size
	 *  · 7: Node by node - Gain ratio (Set of samples) x Size
	 * (Default: Size)<pre>
	 * 
	 * <pre>-PCTB-HC<br>
	 * Build the partial tree ordered by a criteria using Hill Climbing (instead of Best-first).
	 * (Default: true)<pre>
	 *
	 * <pre>-PCTB-CU <br>
	 * Use unpruned partial Consolidated Tree (CT).
	 * (Default: true)<pre>
	 * 
	 * <pre>-PCTB-CC <br>
	 * Collapse the partial Consolidated Tree (CT).
	 * (Default: true)<pre>
	 * 
	 * <pre>-PCTB-WP<br>
	 * Determines whether to prune the base trees without preserving the structure of
	 * the partially consolidated tree (or not). 
	 * (Default: false)<pre>
	 * 
	 * <pre>-PCTB-V mode
	 * Determines how many base trees will be shown:
	 *  None, only the first ten (if they exist) or all.  
	 * (Default: only the first ten)<pre>
	 * 
	 * <pre>-PCTB-P<br>
	 * Determines whether to show the explanation aggregated measures of the all decision trees
	 *  that compose the final classifier (MCS).  
	 * (Default: false)<pre>
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>();

		// J48 and J48Consolidated options
		// ===============================
	    Enumeration<Option> en;
	    en = super.listOptions();
	    while (en.hasMoreElements())
	    	newVector.addElement((Option) en.nextElement());

		// Options to leave partially consolidated the built consolidated tree (PCTB)
		// =========================================================================
		newVector.
		addElement(new Option(
				"\tDetermines the number of inner nodes to leave as consolidated as a percent\n" +
				"\taccording to the inner nodes of the whole consolidated tree (If -PCTB-BP appears(by default).\n" +
				"\tIf -PCTB-BP does not appear, it directly determines the number of internal nodes to be developed\n" +
				"\tas a specific value.\n" +
				"\t(default 20.0)",
				"PCTB-C", 1, "-PCTB-C <consolidation percent (or a specific value)>"));
		newVector.
		addElement(new Option(
				"\tSet the way to set the number of inner nodes to be developed from the partial consolidated tree based on a percentage value\n" +
				"\twith regard to the number of nodes in the whole consolidated tree (instead of using a concrete value)\n",
				"PCTB-BP", 0, "-PCTB-BP"));
		newVector.
		addElement(new Option(
				"\tIndicates the criteria used when choosing the next node to be developed in the construction\n" +
				"\tof the partial consolidated tree:\n" +
				"\t * 0: Original (recursive)\n" +
				"\t * 1: Level by level\n" +
				"\t * 2: Node by node - Pre-order\n" +
				"\t * 3: Node by node - Size\n" +
				"\t * 4: Node by node - Gain ratio (Whole data)\n" +
				"\t * 5: Node by node - Gain ratio (Set of samples)\n" +
				"\t * 6: Node by node - Gain ratio (Whole data) x Size\n" +
				"\t * 7: Node by node - Gain ratio (Set of samples) x Size\n" +
				"\t(default Size)",
				"PCTB-PC", 1, "-PCTB-PC <criteria>"));
		newVector.addElement(new Option("\tSet the heuristic search algorithm to be used with the priority criteria to build the tree as Hill Climbing\n" +
				"\t, instead of Best-first (original).", "PCTB-HC", 0, "-PCTB-HC"));
	    newVector.addElement(new Option("\tUse unpruned partial Consolidated Tree.", "PCTB-CU", 0, "-PCTB-CU"));
	    newVector.addElement(new Option("\tCollpase the partial Consolidated Tree.", "PCTB-CC", 0, "-PCTB-CC"));
	    newVector.
	    addElement(new Option(
	            "\tDetermines whether to prune the base trees without preserving the structure " +
	            "\tthe partially consolidated tree.\n" + 
	            "\t(default false)",
	            "PCTB-WP", 0, "-PCTB-WP"));
		newVector.
		addElement(new Option(
				"\tDetermines how many base trees will be shown:\n" +
				"\tNone, only the first ten (if they exist) or all.\n" +
				"\t(default: only the first ten)",
				"PCTB-V", 1, "-PCTB-V <mode>"));
	    newVector.
	    addElement(new Option(
	            "\tDetermines whether to show the explanation aggregated measures of the all base trees" +
	            "\tthat compose the final classifier (MCS).\n" + 
	            "\t(default false)",
	            "PCTB-P", 0, "-PCTB-P"));

	    return newVector.elements();
	}
	/**
	 * Parses a given list of options.
	 * 
   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * Options to set the Resampling Method (RM) for the generation of samples
	 *  to use in the consolidation process
	 * ============================================================================ 
	 * <pre> -RM-C
	 *  Determines that the way to set the number of samples to be generated will be based on
	 *  a coverage value as a percent. In the case this option is not set, the number of samples
	 *  will be determined using a fixed value. 
	 *  (Set by default)</pre>
	 * 
	 * <pre> -RM-N &lt;number of samples&gt;
	 *  Number of samples to be generated for use in the construction of the consolidated tree.
	 *  It can be set as a fixed value or based on a coverage value as a percent, when -RM-C option
	 *  is used, which guarantees the number of samples necessary to cover adequately the examples 
	 *  of the original sample
	 *  (default 5 for a fixed value or 99% for the case based on a coverage value)</pre>
	 * 
	 * <pre> -RM-R
	 *  Determines whether or not replacement is used when generating the samples.
	 *  (default true)</pre>
	 * 
	 * <pre> -RM-B &lt;Size of each sample(&#37;)&gt;
	 *  Size of each sample(bag), as a percentage of the training set size.
	 *  Combined with the option &lt;distribution minority class&gt; accepts:
	 *  * -1 (sizeOfMinClass): The size of the minority class  
	 *  * -2 (maxSize): Maximum size taking into account &lt;distribution minority class&gt;
	 *  *           and using no replacement
	 *  (default -2(maxSize))</pre>
	 * 
	 * <pre> -RM-D &lt;distribution minority class&gt;
	 *  Determines the new value of the distribution of the minority class, if we want to change it.
	 *  It can be one of the following values:
	 *  * A value between 0 and 100 to change the portion of minority class instances in the new samples
	 *    (If the dataset is multi-class, only the special value 50.0 will be accepted to balance the classes)
	 *  * -1 (free): Works with the instances without taking into account their class  
	 *  * -2 (stratified): Maintains the original class distribution in the new samples
	 *  (default 50.0)</pre>
	 * 
	 * Options to leave partially consolidated the built consolidated tree (PCTB)
	 * ============================================================================ 
	 * <pre>-PCTB-C consolidation percent <br>
	 * Determines the number of inner nodes to leave as consolidated as a percent 
	 * according to the inner nodes of the whole consolidated tree (If -PCTB-BP appears(by default).
	 * If -PCTB-BP does not appear, it directly determines the number of internal nodes to be developed
	 * as a specific value.    
	 * (Default: 20.0)<pre>
	 *
	 * <pre>-PCTB-BP<br>
	 * Set the way to set the number of inner nodes to be developed from the partial 
	 * consolidated tree based on a percentage value with regard to the number of nodes 
	 * in the whole consolidated tree (instead of using a concrete value).
	 * (Default: true)<pre>
	 * 
	 * <pre>-PCTB-PC criteria<br>
	 * Indicates the criteria used when choosing the next node to be developed in the construction
	 * of the partial consolidated tree:
	 *  · 0: Original (recursive)
	 *  · 1: Level by level
	 *  · 2: Node by node - Pre-order
	 *  · 3: Node by node - Size
	 *  · 4: Node by node - Gain ratio (Whole data)
	 *  · 5: Node by node - Gain ratio (Set of samples)
	 *  · 6: Node by node - Gain ratio (Whole data) x Size
	 *  · 7: Node by node - Gain ratio (Set of samples) x Size
	 * (Default: Size)<pre>
	 * 
	 * <pre>-PCTB-HC<br>
	 * Build the partial tree ordered by a criteria using Hill Climbing (instead of Best-first).
	 * (Default: true)<pre>
	 *
	 * <pre>-PCTB-CU <br>
	 * Use unpruned partial Consolidated Tree (CT).
	 * (Default: true)<pre>
	 * 
	 * <pre>-PCTB-CC <br>
	 * Collapse the partial Consolidated Tree (CT).
	 * (Default: true)<pre>
	 * 
	 * <pre>-PCTB-WP<br>
	 * Determines whether to prune the base trees without preserving the structure of
	 * the partially consolidated tree (or not). 
	 * (Default: false)<pre>
	 * 
	 * <pre>-PCTB-V mode
	 * Determines how many base trees will be shown:
	 *  None, only the first ten (if they exist) or all.  
	 * (Default: only the first ten)<pre>
	 * 
	 * <pre>-PCTB-P<br>
	 * Determines whether to show the explanation aggregated measures of the all decision trees
	 *  that compose the final classifier (MCS).  
	 * (Default: false)<pre>
	 * 
   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	/** Whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree. */

	public void setOptions(String[] options) throws Exception {
	    
		// Options to leave partially consolidated the built consolidated tree (PCTB)
		// =========================================================================
		String PCTBconsolidationPercentString = Utils.getOption("PCTB-C", options);
		if (PCTBconsolidationPercentString.length() != 0)
			setPCTBconsolidationPercent(Float.valueOf(PCTBconsolidationPercentString));
		else
			setPCTBconsolidationPercent((float)20.0);

		setPCTBconsolidationPercentHowToSet(
	    	new SelectedTag(Utils.getFlag("PCTB-BP", options) ? NumberConsoNodes_Percentage : NumberConsoNodes_Value, TAGS_WAYS_TO_SET_NUMBER_CONSOLIDATED_NODES));

		String PCTBpriorityCriteriaString = Utils.getOption("PCTB-PC", options);
		if (PCTBpriorityCriteriaString.length() != 0)
			setPCTBpriorityCriteria(new SelectedTag(Integer.parseInt(PCTBpriorityCriteriaString), TAGS_WAYS_TO_SET_PRIORITY_CRITERIA));
		else
			setPCTBpriorityCriteria(new SelectedTag(PriorCrit_Size, TAGS_WAYS_TO_SET_PRIORITY_CRITERIA)); // default: Size

		setPCTBheuristicSearchAlgorithm(
		    	new SelectedTag(Utils.getFlag("PCTB-HC", options) ? SearchAlg_HillClimbing : SearchAlg_BestFirst, TAGS_WAYS_TO_SET_SEARCH_ALGORITHM));

		setPCTBunprunedCT(Utils.getFlag("PCTB-CU", options));

		setPCTBcollapseCT(Utils.getFlag("PCTB-CC", options));

		setPCTBpruneBaseTreesWithoutPreservingConsolidatedStructure(Utils.getFlag("PCTB-WP", options));

		String PCTBvisualizeBaseTreesString = Utils.getOption("PCTB-V", options);
		if (PCTBvisualizeBaseTreesString.length() != 0)
			setPCTBvisualizeBaseTrees(new SelectedTag(Integer.parseInt(PCTBvisualizeBaseTreesString), TAGS_VISUALIZE_BASE_TREES));
		else
			setPCTBvisualizeBaseTrees(new SelectedTag(Visualize_FirstOnes, TAGS_VISUALIZE_BASE_TREES)); // default: only the first ten

		setPCTBprintExplanationMeasuresBaseTrees(Utils.getFlag("PCTB-P", options));

	    // J48 and J48Consolidated options
		// ===============================
	    super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String [] getOptions() {

		Vector<String> result = new Vector<String>();

		// J48 and J48Consolidated options
		// ===============================
		String[] options = super.getOptions();
		for (int i = 0; i < options.length; i++)
			result.add(options[i]);

		// Options to leave partially consolidated the built consolidated tree (PCTB)
		// =========================================================================
		result.add("-PCTB-C");
		result.add("" + m_PCTBconsolidationPercent);

		if (m_PCTBconsolidationPercentHowToSet == NumberConsoNodes_Percentage)
			result.add("-PCTB-BP");

		result.add("-PCTB-PC");
		result.add("" + m_PCTBpriorityCriteria);
		
		if (m_PCTBheuristicSearchAlgorithm == SearchAlg_HillClimbing)
			result.add("-PCTB-HC");

		if (m_PCTBunprunedCT) {
			result.add("-PCTB-CU");
		}

		if (m_PCTBcollapseCT) {
			result.add("-PCTB-CC");
		}

		if (m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure)
			result.add("-PCTB-WP");

		result.add("-PCTB-V");
		result.add("" + m_PCTBvisualizeBaseTrees);

		if (m_PCTBprintExplanationMeasuresBaseTrees)
			result.add("-PCTB-P");

		return (String[]) result.toArray(new String[result.size()]);	  
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier
	 */
	public String toString() {
		String st;
		st = "J48PartiallyConsolidated-Bagging classifier\n";
		// Add a separator
		char[] ch_line = new char[st.length()];
		for (int i = 0; i < ch_line.length; i++)
			ch_line[i] = '-';
		String line = String.valueOf(ch_line);
		line += "\n";
		st += "Priority criteria to grow the partial consolidated tree: ";
		switch(m_PCTBpriorityCriteria) {
			case PriorCrit_Original:
				st += "Original (recursive)";break;
			case PriorCrit_Levelbylevel:
				st += "Level by level";break;
			case PriorCrit_Preorder:
				st += "Node by node - Preorder";break;
			case PriorCrit_Size:
				st += "Node by node - Size";break;
			case PriorCrit_GainratioWholeData:
				st += "Node by node - Gain ratio (Whole data)";break;
			case PriorCrit_GainratioSetSamples:
				st += "Node by node - Gain ratio (Set of samples)";break;
			case PriorCrit_GainratioWholeData_Size:
				st += "Node by node - Gain ratio (Whole data) weighted * Size";break;
			case PriorCrit_GainratioSetSamples_Size:
				st += "Node by node - Gain ratio (Set of samples) weighted * Size";break;
		}
		st += "\n";
		if((m_PCTBpriorityCriteria >= PriorCrit_Size) && (m_PCTBpriorityCriteria <= PriorCrit_GainratioSetSamples_Size)) {
			if(m_PCTBheuristicSearchAlgorithm == SearchAlg_HillClimbing)
				st += " using Hill Climbing as heuristic search algorithm\n";
			else
				st += " using Best-first as heuristic search algorithm\n";
		}
		if (m_PCTBconsolidationPercentHowToSet == NumberConsoNodes_Percentage) {
			if (m_PCTBpriorityCriteria == PriorCrit_Levelbylevel) {
				st += "Consolidation percent (in terms of number of levels of the tree) = " + Utils.doubleToString(m_PCTBconsolidationPercent,2) + "%";
				if (m_root != null) {
					int numberNodesConso = ((C45PartiallyConsolidatedPruneableClassifierTree)m_root).getNumInternalNodesConso();
					if (numberNodesConso >= 0)
						st += " => Number of levels to grow: " + numberNodesConso;
				}
			} else {
				st += "Consolidation percent = " + Utils.doubleToString(m_PCTBconsolidationPercent,2) + "%";
				if (m_root != null) {
					int numberNodesConso = ((C45PartiallyConsolidatedPruneableClassifierTree)m_root).getNumInternalNodesConso();
					if (numberNodesConso >= 0)
						st += " => Internal nodes to grow: " + numberNodesConso;
				}
			}
			st += "\n";
		} else { // ConsolidationNumber_Value
			if (m_PCTBpriorityCriteria == PriorCrit_Levelbylevel)
				st += "Number of levels of the partial consolidated tree to grow = " + Utils.doubleToString(m_PCTBconsolidationPercent,0) + "\n";
			else
				st += "Number of inner nodes of the partial consolidated tree to grow = " + Utils.doubleToString(m_PCTBconsolidationPercent,0) + "\n";
		}
		if (m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure)
			st += "·Without preserving the structure of the partially consolidated tree in the base trees\n";
		else
			st += "·Preserving the structure of the partially consolidated tree in the base trees\n";
		st += line;
		//st += super.toString();
		if (m_root == null)
			st += "No classifier built";
		else {
			st += "J48Consolidated " + (m_PCTBunprunedCT? "unpruned " : "") + (m_PCTBcollapseCT? "(collapsed) " : "") + "tree\n" + 
					toStringResamplingMethod() + m_root.toString();
			st += toStringVisualizeBaseTrees(line);
			st += toStringPrintExplanationMeasuresMCS();
			if(m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure) {
				st += "\n---------------------------------------------------"
					+ "\nMeasures to evaluate the effect of pruning base trees"
					+ "\nwithout preserving the structure of the partial consolidated tree:"
					+ "\nGiven, at each node of the partial consolidated tree, the percentage"
					+ "\nof base trees containing the same split division [Str: dd.dd%], we obtain:"
					+ "\n· Mean: " + Utils.roundDouble(measureAvgPercBaseTreesPreservingStructure(),2) + "%"
					+ "\n· Minimum: " + Utils.roundDouble(measureMinPercBaseTreesPreservingStructure(),2) + "%"
					+ "\n· Maximum: " + Utils.roundDouble(measureMaxPercBaseTreesPreservingStructure(),2) + "%"
					+ "\n· Median: " + Utils.roundDouble(measureMdnPercBaseTreesPreservingStructure(),2) + "%"
					+ "\n· Std.Dev.: " + Utils.roundDouble(measureDevPercBaseTreesPreservingStructure(),2) + "%"
					+ "\n---------------------------------------------------";
			}
		}
		return st;
	}

	/**
	 * Returns the base trees as text
	 * 
	 * @return the base trees as text
	 */
	public String toStringVisualizeBaseTrees(String line) {
		String st = "";
		if (m_PCTBvisualizeBaseTrees > Visualize_None) {
			/** Base tree vector */
			if (m_Classifiers != null){
				int maxBaseTrees;
				st += line;
				if ((m_PCTBvisualizeBaseTrees == Visualize_All) || (m_Classifiers.length <= 10)) {
					maxBaseTrees = m_Classifiers.length;
					st += "Set of " + m_Classifiers.length + " base " + (m_unpruned? "unpruned " : "") + "trees:\n";
				}
				else {
					maxBaseTrees = 10;
					st += "Set of " + m_Classifiers.length + " base " + (m_unpruned? "unpruned " : "") + "trees (*Only the first ten!):\n";
				}
				for (int iSample = 0; iSample < maxBaseTrees; iSample++){
					st += line;
					st += iSample + "-th base tree:\n";
					st += m_Classifiers[iSample].toString();
				}
			}
		}
		return st;
	}

	/**
	 * Returns a description of the explanation measures of the all decision trees
	 *  that compose the final classifier (MCS)
	 * 
	 * @return a description of the explanation measures of all DTs
	 */
	public String toStringPrintExplanationMeasuresMCS() {
		String st = "";
		if (m_PCTBprintExplanationMeasuresBaseTrees) {
			st += "\n--- Complexity/Explanation aggregated measures  ---"
				+ "\n---  of the whole multiple classifier           ---"
				+ "\n---------------------------------------------------\n";
			
			String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
			ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
			String[] stTreeMeasures = new String[]{"NumLeaves", /*"NumRules",*/ "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
			ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));

			for(String ms: treeMeasures) {
				st += ms + ": ";
				for(String op: metaOperations) {
					String mtms = "measure" + op + ms;
					st += op + " = " + Utils.roundDouble(getMeasure(mtms),2) + " ";
				}
				st += "\n";
			}
		}
		return st;
	}

	/**
	 * Returns the meta-classifier as Java source code.
	 * (based on toSource() function of AdaBoostM1 class)
	 *
	 * @see weka.classifiers.meta.AdaBoostM1.toSource(String)
	 *
	 * @param className the classname of the generated code
	 * @return the tree as Java source code
	 * @throws Exception if something goes wrong
	 */
	public String toSource(String className) throws Exception {
		if (m_Classifiers == null) {
			throw new Exception("No model built yet");
		}
		Instances data = m_root.getTrainingData();
		int numClasses = data.numClasses();

		/** Number of Samples. */
		int numberSamples = m_Classifiers.length;
		StringBuffer text = new StringBuffer("class ");
		text.append(className).append(" {\n\n");

		text.append("  public static double classify(Object[] i) throws Exception {\n");

		if (numberSamples == 1) {
			text.append("    return " + className + "_0.classify(i);\n");
		} else {
			text.append("    double [] sums = new double [" + numClasses + "];\n");
			for (int i = 0; i < numberSamples; i++) {
				text.append("    sums[(int) " + className + '_' + i
						+ ".classify(i)] += (double)1 ;\n");
			}
			text.append("    double maxV = sums[0];\n" + "    int maxI = 0;\n"
					+ "    for (int j = 1; j < " + numClasses + "; j++) {\n"
					+ "      if (sums[j] > maxV) { maxV = sums[j]; maxI = j; }\n"
					+ "    }\n    return (double) maxI;\n");
		}
		text.append("  }\n}\n");

		for (int i = 0; i < m_Classifiers.length; i++) {
			/**
			 * Based on toSource() function of J48 class
			 *
			 * @see weka.classifiers.trees.J48.toSource(String) 
			 */
			StringBuffer[] sourceTree = ((ClassifierTree) m_Classifiers[i]).toSource(className + '_' + i);
			String sourceClas = "class " + className + '_' + i + " {\n\n"
				      + "  public static double classify(Object[] i)\n"
				      + "    throws Exception {\n\n" + "    double p = Double.NaN;\n"
				      + sourceTree[0] // Assignment code
				      + "    return p;\n" + "  }\n" + sourceTree[1] // Support code
				      + "}\n";
			text.append(sourceClas);
		}
		return text.toString();
	}
	
	/**
	 * Returns the time taken to build the whole Consolidated Tree (CT),
	 * including pruning/collapsing, if required.
	 * 
	 * @return elapsed time
	 */
	public double measureElapsedTimeTrainingWholeCT() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getElapsedTimeTrainingWholeCT();
	}
	
	/**
	 * Returns the time taken to build the partial Consolidated Tree (CT),
	 * including pruning/collapsing, if required.
	 * 
	 * @return elapsed time
	 */
	public double measureElapsedTimeTrainingPartialCT() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getElapsedTimeTrainingPartialCT();
	}

	/**
	 * Returns the time taken to build all base trees that compose the final multiple classifier (Bagging),
	 * including pruning/collapsing, if required.
	 * 
	 * @return elapsed time
	 */
	public double measureElapsedTimeTrainingAssocBagging() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getElapsedTimeTrainingAssocBagging();
	}

	/**
	 * Returns the average percentage of base trees preserving structure throughout the tree
	 * 
	 * @return average percentage
	 */
	public double measureAvgPercBaseTreesPreservingStructure() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getAvgPercBaseTreesPreservingStructure();
	}

	/**
	 * Returns the minimum percentage of base trees preserving structure throughout the tree
	 * 
	 * @return minimum percentage
	 */
	public double measureMinPercBaseTreesPreservingStructure() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getMinPercBaseTreesPreservingStructure();
	}

	/**
	 * Returns the maximum percentage of base trees preserving structure throughout the tree
	 * 
	 * @return maximum percentage
	 */
	public double measureMaxPercBaseTreesPreservingStructure() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getMaxPercBaseTreesPreservingStructure();
	}

	/**
	 * Returns the median percentage of base trees preserving structure throughout the tree
	 * 
	 * @return median percentage
	 */
	public double measureMdnPercBaseTreesPreservingStructure() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getMdnPercBaseTreesPreservingStructure();
	}

	/**
	 * Returns the standard deviation percentage of base trees preserving structure throughout the tree
	 * 
	 * @return std deviation percentage
	 */
	public double measureDevPercBaseTreesPreservingStructure() {
		return ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_root).getDevPercBaseTreesPreservingStructure();
	}

	/**
	 * Returns an enumeration of the additional measure names
	 * (Added also those produced by the base algorithm).
	 * 
	 * @return an enumeration of the measure names
	 */
	@Override
	public Enumeration<String> enumerateMeasures() {
		Vector<String> newVector = new Vector<String>(1);
		newVector.addAll(Collections.list(super.enumerateMeasures()));
		String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
		ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
		String[] stTreeMeasures = new String[]{"NumLeaves", "NumRules", "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
		ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));

		for(String op: metaOperations)
			for(String ms: treeMeasures)
				newVector.addElement("measure" + op + ms);

		newVector.addElement("measureElapsedTimeTrainingWholeCT");
		newVector.addElement("measureElapsedTimeTrainingPartialCT");
		newVector.addElement("measureElapsedTimeTrainingAssocBagging");
		newVector.addElement("measureAvgPercBaseTreesPreservingStructure");
		newVector.addElement("measureMinPercBaseTreesPreservingStructure");
		newVector.addElement("measureMaxPercBaseTreesPreservingStructure");
		newVector.addElement("measureMdnPercBaseTreesPreservingStructure");
		newVector.addElement("measureDevPercBaseTreesPreservingStructure");

		return newVector.elements();
	}

	/**
	 * 	Returns the value of a named aggregate measure associated to
	 *  a meta-classifier based on a set of decision trees.
	 *
	 * @param singleMeasureName the name of the measure to query for its value
	 * @param iop identifies the type of aggregated operation
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	public double getMetaAggregateMeasure(String singleMeasureName, int iop) {
		double res = (double)0.0;
		double[] vValues = new double[m_Classifiers.length];
		for(int i = 0; i < m_Classifiers.length; i++) {
			vValues[i] = ((AdditionalMeasureProducer)(m_Classifiers[i])).getMeasure(singleMeasureName);
		}
		switch (iop) {
			case 0: // Avg
				res = Utils.mean(vValues); break;
			case 1: // Min
				res = vValues[Utils.minIndex(vValues)]; break;
			case 2: // Max
				res = vValues[Utils.maxIndex(vValues)]; break;
			case 3: // Sum
				res = Utils.sum(vValues); break;
			case 4: // Mdn
				res = Utils.kthSmallestValue(vValues, m_Classifiers.length / 2); break;
			case 5: // Dev 
				res = Math.sqrt(Utils.variance(vValues)); break;
		}
		return res;
	}

	/**
	 * Returns the value of the named measure.
	 *
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	@Override
	public double getMeasure(String additionalMeasureName) {
		if(Collections.list(new J48Consolidated().enumerateMeasures()).contains(additionalMeasureName)) {
			return super.getMeasure(additionalMeasureName);
		} else {
			String[] stMetaOperations = new String[]{"Avg", "Min", "Max", "Sum", "Mdn", "Dev"};
			ArrayList<String> metaOperations = new ArrayList<>(Arrays.asList(stMetaOperations));
			String[] stTreeMeasures = new String[]{"NumLeaves", "NumRules", "NumInnerNodes", "ExplanationLength", "WeightedExplanationLength"};
			ArrayList<String> treeMeasures = new ArrayList<>(Arrays.asList(stTreeMeasures));
	
			// If it is an aggregate measure associated to the meta-classifier...
			for(String op: metaOperations)
				for(String ms: treeMeasures) {
					String mtms = "measure" + op + ms;
					if (additionalMeasureName.equalsIgnoreCase(mtms)) {
						String sms = "measure" + ms;
						int iop = metaOperations.indexOf(op);
						return getMetaAggregateMeasure(sms, iop);
					}
				}
			
			if (additionalMeasureName.compareToIgnoreCase("measureElapsedTimeTrainingWholeCT") == 0) {
				return measureElapsedTimeTrainingWholeCT();
			} else if (additionalMeasureName.compareToIgnoreCase("measureElapsedTimeTrainingPartialCT") == 0) {
				return measureElapsedTimeTrainingPartialCT();
			} else if (additionalMeasureName.compareToIgnoreCase("measureElapsedTimeTrainingAssocBagging") == 0) {
				return measureElapsedTimeTrainingAssocBagging();
			} else if (additionalMeasureName.compareToIgnoreCase("measureAvgPercBaseTreesPreservingStructure") == 0) {
				return measureAvgPercBaseTreesPreservingStructure();
			} else if (additionalMeasureName.compareToIgnoreCase("measureMinPercBaseTreesPreservingStructure") == 0) {
				return measureMinPercBaseTreesPreservingStructure();
			} else if (additionalMeasureName.compareToIgnoreCase("measureMaxPercBaseTreesPreservingStructure") == 0) {
				return measureMaxPercBaseTreesPreservingStructure();
			} else if (additionalMeasureName.compareToIgnoreCase("measureMdnPercBaseTreesPreservingStructure") == 0) {
				return measureMdnPercBaseTreesPreservingStructure();
			} else if (additionalMeasureName.compareToIgnoreCase("measureDevPercBaseTreesPreservingStructure") == 0) {
				return measureDevPercBaseTreesPreservingStructure();
			} else
				throw new IllegalArgumentException(additionalMeasureName 
					+ " not supported (J48PartiallyConsolidated)");
		}
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBconsolidationPercentTipText() {
		return "Consolidation percent (or number of inner nodes) that determines the number of nodes of the consolidated tree to be developed.";
	}

	/**
	 * Get the value of ConsolidationPercent (if PCTBnumberConsoNodesHowToSet = "based on a pertentage value (%)") or
	 * directly the number of internal nodes (if PCTBnumberConsoNodesHowToSet = "using a specific value")
	 *
	 * @return Value of ConsolidationPercent or number of nodes.
	 */
	public float getPCTBconsolidationPercent() {

		return m_PCTBconsolidationPercent;
	}

	/**
	 * Set the value of Consolidation Percent or directly the number of internal nodes
	 *
	 * @param v Value to assign to ConsolidationPercent or number of nodes.
	 * @throws Exception if an option is not supported
	 */
	public void setPCTBconsolidationPercent(float v) throws Exception {
		if (m_PCTBconsolidationPercentHowToSet ==  NumberConsoNodes_Percentage) {
			if ((v < 0) || (v > 100))
				throw new Exception("The consolidation percent (%) has to be a value greater than or equal to zero and smaller " +
						"than or equal to 100!");
		}
		m_PCTBconsolidationPercent = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBconsolidationPercentHowToSetTipText() {
		return "Way to set the number of nodes to be developed from the partial consolidated tree:\n" +
				" · using a fixed value which directly indicates the number nodes to consolidate or\n" +
				" · based on a value as a percentage with regard to the number of nodes in the whole\n" +
				"   consolidated tree.\n";
	}
	
	/**
	 * Get the value of PCTBconsolidationPercentHowToSet.
	 *
	 * @return Value of PCTBconsolidationPercentHowToSet.
	 */
	public SelectedTag getPCTconsolidationPercentHowToSet() {
		return new SelectedTag(m_PCTBconsolidationPercentHowToSet,
				TAGS_WAYS_TO_SET_NUMBER_CONSOLIDATED_NODES);
	}
	
	/**
	 * Set the value of PCTBconsolidationPercentHowToSet. Values other than
	 * NumberConsoNodes_Percentage, or NumberConsoNodes_Value will be ignored.
	 *
	 * @param newWayToSetnumberConsoNodes the way to set the number of samples to use
	 * @throws Exception if an option is not supported
	 */
	public void setPCTBconsolidationPercentHowToSet(SelectedTag newWayToSetnumberConsoNodes) throws Exception {
		if (newWayToSetnumberConsoNodes.getTags() == TAGS_WAYS_TO_SET_NUMBER_CONSOLIDATED_NODES) 
		{
			int newEvWay = newWayToSetnumberConsoNodes.getSelectedTag().getID();

			if (newEvWay == NumberConsoNodes_Value || newEvWay == NumberConsoNodes_Percentage)
				m_PCTBconsolidationPercentHowToSet = newEvWay;
			else 
				throw new IllegalArgumentException("Wrong selection type, value should be: "
						+ "between 1 and 2");
		}
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String PCTBpriorityCriteriaTipText() {
		return "Build the partial consolidated tree driven by a criteria:\n" +
				" · Original (recursive)\n" +
				" · Level by level\n" +
				" · Node by node: Pre-order\n" +
				" · Node by node: Size\n" +
				" · Node by node: Gain ratio (Whole data)\n" + 
				" · Node by node: Gain ratio (Set of samples)\n" + 
				" · Node by node: Gain ratio (Whole data) x Size\n" +
				" · Node by node: Gain ratio (Set of samples) x Size\n";
	}

	/**
	 * Get the value of PCTBpriorityCriteria.
	 * 
	 * @return Value of PCTBpriorityCriteria.
	 */
	public SelectedTag getPCTBpriorityCriteria() {
		return new SelectedTag(m_PCTBpriorityCriteria,
				TAGS_WAYS_TO_SET_PRIORITY_CRITERIA);
	}

	/**
	 * Set the value of PCTBpriorityCriteria.
	 * 
	 * @param v Value to assign to PCTBpriorityCriteria.
	 */

	public void setPCTBpriorityCriteria(SelectedTag newPriorityCriteria) throws Exception {
		if (newPriorityCriteria.getTags() == TAGS_WAYS_TO_SET_PRIORITY_CRITERIA) 
		{
			int newPriority = newPriorityCriteria.getSelectedTag().getID();

			if (newPriority == PriorCrit_Original || newPriority == PriorCrit_Levelbylevel || newPriority == PriorCrit_Preorder || newPriority == PriorCrit_Size || 
				newPriority == PriorCrit_GainratioWholeData || newPriority == PriorCrit_GainratioWholeData_Size ||
				newPriority == PriorCrit_GainratioSetSamples || newPriority == PriorCrit_GainratioSetSamples_Size)
				m_PCTBpriorityCriteria = newPriority;
			else 
				throw new IllegalArgumentException("Wrong selection type, value should be: "
						+ "between 0 and 7");
		}
	}
	
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBheuristicSearchAlgorithmTipText() {
		return "Way to set the heuristic search algorithm to be used with the priority criteria to build the partial consolidated tree:\n" +
				" * Best-first (Original)\n" +
				" * Hill Climbing (In-depth)\n";
	}
	
	/**
	 * Get the value of PCTBheuristicSearchAlgorithm.
	 *
	 * @return Value of PCTBheuristicSearchAlgorithm.
	 */
	public SelectedTag getPCTBheuristicSearchAlgorithm() {
		return new SelectedTag(m_PCTBheuristicSearchAlgorithm,
				TAGS_WAYS_TO_SET_SEARCH_ALGORITHM);
	}

	/**
	 * Set the value of PCTBheuristicSearchAlgorithm. Values other than
	 * BestFirst, or HillClimbing will be ignored.
	 *
	 * @param newWayToSetSearchAlgorithm the way to set the heuristic search algorithm to use
	 * @throws Exception if an option is not supported
	 */
	public void setPCTBheuristicSearchAlgorithm(SelectedTag newWayToSetSearchAlgorithm) throws Exception {
		if (newWayToSetSearchAlgorithm.getTags() == TAGS_WAYS_TO_SET_SEARCH_ALGORITHM) 
		{
			int newEvWay = newWayToSetSearchAlgorithm.getSelectedTag().getID();

			if (newEvWay == SearchAlg_BestFirst || newEvWay == SearchAlg_HillClimbing)
				m_PCTBheuristicSearchAlgorithm = newEvWay;
			else 
				throw new IllegalArgumentException("Wrong selection type, value should be: "
						+ "BestFirst(0) or HillClimbing(1)");
		}
	}
	
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBunprunedCTTipText() {
		return "whether to prune or not the partial Consolidated Tree (CT).";
	}
	
	/**
	 * Set whether to prune or not the partial Consolidated Tree (CT). 
	 *
	 * @param unprunedCT whether to prune or not the partial Consolidated Tree (CT).
	 */
	public void setPCTBunprunedCT(boolean unprunedCT) {

		m_PCTBunprunedCT = unprunedCT;
	}

	/**
	 * Get whether to prune or not the partial Consolidated Tree (CT).
	 *
	 * @return whether to prune or not the partial Consolidated Tree (CT).
	 */
	public boolean getPCTBunprunedCT() {

		return m_PCTBunprunedCT;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBcollapseCTTipText() {
		return "whether to collapse or not the partial Consolidated Tree (CT).";
	}
	
	/**
	 * Set whether to collapse or not the partial Consolidated Tree (CT). 
	 *
	 * @param unprunedCT whether to collapse or not the partial Consolidated Tree (CT).
	 */
	public void setPCTBcollapseCT(boolean collapseCT) {

		m_PCTBcollapseCT = collapseCT;
	}

	/**
	 * Get whether to collapse or not the partial Consolidated Tree (CT).
	 *
	 * @return whether to collapse or not the partial Consolidated Tree (CT).
	 */
	public boolean getPCTBcollapseCT() {

		return m_PCTBcollapseCT;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBpruneBaseTreesWithoutPreservingConsolidatedStructureTipText() {
		return "Whether to prune the base trees without preserving the structure of the partially consolidated tree.";
	}

	/**
	 * Set whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree. 
	 *
	 * @param notPreserving whether to prune without preserving the consolidated structure.
	 */
	public void setPCTBpruneBaseTreesWithoutPreservingConsolidatedStructure(boolean notPreserving) {

		m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure = notPreserving;
	}

	/**
	 * Get whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree.
	 *
	 * @return whether to prune without preserving the consolidated structure
	 */
	public boolean getPCTBpruneBaseTreesWithoutPreservingConsolidatedStructure() {

		return m_PCTBpruneBaseTreesWithoutPreservingConsolidatedStructure;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBvisualizeBaseTreesTipText() {
		return "Mode to visualize the set of base trees that compose the final classifier (MCS): None, only the first ten or all";
	}

	/**
	 * Get the value of visualizeBaseTrees.
	 *
	 * @return Value of visualizeBaseTrees.
	 */
	public SelectedTag getPCTBvisualizeBaseTrees() {
	    return new SelectedTag(m_PCTBvisualizeBaseTrees,
	    		TAGS_VISUALIZE_BASE_TREES);
	}

	/**
	 * Set the value of visualizeBaseTrees.
	 *
	 * @param mode Value to assign to visualizeBaseTrees.
	 * @throws Exception if an option is not supported
	 */
	public void setPCTBvisualizeBaseTrees(SelectedTag mode) throws Exception {
	  	if (mode.getTags() == TAGS_VISUALIZE_BASE_TREES) 
	  	{
	  		int newMode = mode.getSelectedTag().getID();
	  		
		    if (newMode == Visualize_None || newMode == Visualize_FirstOnes || newMode == Visualize_All)
		    	m_PCTBvisualizeBaseTrees = newMode;
		    else 
		    	throw new IllegalArgumentException("Wrong selection type, value should be: "
		                                           + "between 1 and 3");
		 }
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String PCTBprintExplanationMeasuresBaseTreesTipText() {
		return "Whether to show the explanation aggregated measures of all base trees that compose the final classifier (MCS).";
	}
	
	/**
	 * Set whether to show the explanation aggregated measures of all base trees. 
	 *
	 * @param printExplanationMeasures whether to show the explanation measures of all DTs.
	 */
	public void setPCTBprintExplanationMeasuresBaseTrees(boolean printExplanationMeasures) {

		m_PCTBprintExplanationMeasuresBaseTrees = printExplanationMeasures;
	}

	/**
	 * Get whether to show the explanation aggregated measures of all base trees.
	 *
	 * @return whether to to show the explanation measures
	 */
	public boolean getPCTBprintExplanationMeasuresBaseTrees() {

		return m_PCTBprintExplanationMeasuresBaseTrees;
	}

}