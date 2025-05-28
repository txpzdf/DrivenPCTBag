/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    DatasetCharacteristicsExtractor.java
 *    Copyright (C) 2024 ALDAPA Team (http://www.aldapa.eus)
 *    Faculty of Informatics, Donostia, 20018
 *    University of the Basque Country (UPV/EHU), Basque Country
 *    
 */
package weka.classifiers.rules;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Sourcable;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * <!-- globalinfo-start -->
 * Class for extracting the main descriptive characteristics of a dataset based on WEKA's simplest classifier, ZeroR.<p/>
 * When used as a classification algorithm in the WEKA's Experimenter, it returns the descriptive features (number of classes, number of attributes...) of a set of datasets as if they were metrics (Comparison field) used to evaluate the goodness of the classifier (like Percent_correct, Area_under_ROC, Elapsed_Time_training...).<p/>
 * For proper results configuration (Setup tab of the Experimenter), it's recommended to set the 'Experiment Type' to "Train/Test Percentage Split (order preserved)" with 100% 'Train Percentage'.
 * This ensures measures like Number_of_training_instances or NumMissingValuesDataset aren't affected by Train/Test data splits of the default 'Cross-validation' option.<p/>
 * To obtain research-ready results, specify 'CSV file' as 'Results Destination' and provide a filename. After running the experiment, the generated CSV can be opened in spreadsheet software, displaying datasets in rows and their complete features (plus ZeroR metrics) in columns - similar to the dataset description tables commonly found in machine learning publications.<p/>
 * <br/> 
 * List of extracted characteristics (all starting with “measure” due to WEKA naming convention):<p/> 
 * <ul>
 * <li><b>NumAttributes</b>: Number of attributes of the dataset (without class)</li>
 * <li><b>NumNumericAttributes</b>: Number of numeric attributes of the dataset (without class)</li>
 * <li><b>NumNominalAttributes</b>: Number of nominal attributes of the dataset (without class)</li>
 * <li><b>MissingValues</b>: Whether there are missing values in the dataset (1.0) or not (0.0) (without class)</li>
 * <li><b>NumAttsMissingValues</b>: Number of attributes with missing values (without class)</li>
 * <li><b>NumMissingValuesDataset</b>: Number of examples with missing values in the dataset (without class)</li>
 * <li><b>PercentMissingValuesDataset</b>: Percentage of examples with missing values in the dataset (without class)</li>
 * <li><b>NumClasses</b>: Number of classes in the dataset</li>
 * <li><b>EmptyClass</b>: Whether there are any empty classes (1.0) or not (0.0)</li>
 * <li><b>NumFirstClass</b>: Number of examples of the first class (considered by WEKA as positive by default)</li>
 * <li><b>MinClassIndex</b>: Index of minority class (discarding empty classes)</li>
 * <li><b>NumMinClass</b>: Number of examples of minority class (discarding empty classes)</li>
 * <li><b>PercentMinClass</b>: Percentage of examples of minority class (discarding empty classes)</li>
 * <li><b>NumMajClass</b>: Number of examples of majority class</li>
 * <li><b>PercentMajClass</b>: Percentage of examples of majority class</li>
 * </ul>
 * This class was used in the following paper where an extensive experimentation was carried out with 96 different datasets:<br/>
 * Jes&uacute;s M. P&eacute;rez and Olatz Arbelaitz.  
 * "Multi-Criteria Node Selection in Direct PCTBagging: Balancing Interpretability and Accuracy with Bootstrap Sampling and Unrestricted Pruning". Information Sciences (2025), in Press.
 * <a href="https://doi.org/10.1016/j.ins.2025.XX.XXX" target="_blank">doi:10.1016/j.ins.2025.XX.XXX</a>
 * <p/>
 * <!-- globalinfo-end -->
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Perez2025,
 *    title = "Multi-Criteria Node Selection in Direct PCTBagging: Balancing Interpretability and Accuracy with Bootstrap Sampling and Unrestricted Pruning",
 *    journal = "Information Sciences (in Press)",
 *    volume = "",
 *    number = "",
 *    pages = "1 - X",
 *    year = "2025",
 *    doi = "10.1016/j.ins.2025.XX.XXX",
 *    author = "Jes\'us M. P\'erez and Olatz Arbelaitz"
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * *************************************************************************************<br/>
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * ZeroR options <br/>
 * =============
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus)
 * @version $Revision: 1.1 $
 */
public class DatasetCharacteristicsExtractor
	extends ZeroR
	implements WeightedInstancesHandler, Sourcable, TechnicalInformationHandler, AdditionalMeasureProducer, Summarizable {

	/** for serialization */
	private static final long serialVersionUID = -5996036009128552856L;

	private Instances m_data;

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the explorer/experimenter
	 *         gui
	 */
	public String globalInfo() {
		return "Class for extracting the main descriptive characteristics of a dataset based on WEKA's simplest classifier, ZeroR.\n"
				+ "When used as a classification algorithm in the WEKA's Experimenter, it returns the descriptive features "
				+ "(number of classes, number of attributes...) of a set of datasets as if they were metrics (Comparison field) "
				+ "used to evaluate the goodness of the classifier (like Percent_correct, Area_under_ROC, Elapsed_Time_training...).\n\n"
				
				+ "For proper results configuration (Setup tab of the Experimenter), it's recommended to set the 'Experiment Type' "
				+ "to \"Train/Test Percentage Split (order preserved)\" with 100% 'Train Percentage'.\n\n"
				
				+ "This ensures measures like Number_of_training_instances or NumMissingValuesDataset aren't affected by Train/Test "
				+ "data splits of the default 'Cross-validation' option.\n\n"
				
				+ "To obtain research-ready results, specify 'CSV file' as 'Results Destination' and provide a filename. After running "
				+ "the experiment, the generated CSV can be opened in spreadsheet software, displaying datasets in rows and their "
				+ "complete features (plus ZeroR metrics) in columns - similar to the dataset description tables commonly found in "
				+ "machine learning publications.\n\n"
				
				+ "List of extracted characteristics (all starting with “measure” due to WEKA naming convention):\n"
				+ " · NumAttributes: Number of attributes of the dataset (without class)\n"
				+ " · NumNumericAttributes: Number of numeric attributes of the dataset (without class)\n"
				+ " · NumNominalAttributes</b>: Number of nominal attributes of the dataset (without class)\n"
				+ " · MissingValues: Whether there are missing values in the dataset (1.0) or not (0.0) (without class)\n"
				+ " · NumAttsMissingValues: Number of attributes with missing values (without class)\n"
				+ " · NumMissingValuesDataset: Number of examples with missing values in the dataset (without class)\n"
				+ " · PercentMissingValuesDataset: Percentage of examples with missing values in the dataset (without class)\n"
				+ " · NumClasses: Number of classes in the dataset\n"
				+ " · EmptyClass: Whether there are any empty classes (1.0) or not (0.0)\n"
				+ " · NumFirstClass: Number of examples of the first class (considered by WEKA as positive by default)\n"
				+ " · MinClassIndex: Index of minority class (discarding empty classes)\n"
				+ " · NumMinClass: Number of examples of minority class (discarding empty classes)\n"
				+ " · PercentMinClass: Percentage of examples of minority class (discarding empty classes)\n"
				+ " · NumMajClass: Number of examples of majority class\n"
				+ " · PercentMajClass: Percentage of examples of majority class\n\n"
				
				+ "This class was used in the following paper where an extensive experimentation was carried out with 96 different datasets:\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Jesús M. Pérez and Olatz Arbelaitz");
		result.setValue(Field.YEAR, "2025");
		result.setValue(Field.TITLE, "Multi-Criteria Node Selection in Direct PCTBagging: Balancing Interpretability and Accuracy with Bootstrap Sampling and Unrestricted Pruning");
	    result.setValue(Field.JOURNAL, "Information Sciences");
	    result.setValue(Field.VOLUME, "");
	    result.setValue(Field.PAGES, "1-XX");
	    result.setValue(Field.URL, "https://doi.org/10.1016/j.ins.2025.XX.XXX");

		return result;
	}

	/**
	 * Generates the ZeroR classifier, associating the training sample with m_data
	 * in order to extract its descriptive characteristics.
	 * 
	 * @param instances set of instances serving as training data
	 * @throws Exception if the classifier has not been generated successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		super.buildClassifier(instances);
		m_data = instances;
	}

	/**
	 * Returns number of attributes of the dataset (without class). 
	 * 
	 * @return number of attributes of the dataset (without class).
	 */
	public double measureNumAttributes() {
		return (double)(m_data.numAttributes() - 1);
	}

	/**
	 * Returns number of numeric attributes of the dataset (without class). 
	 * 
	 * @return number of numeric attributes of the dataset (without class).
	 */
	public double measureNumNumericAttributes() {
		int count = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			Attribute at = m_data.attribute(i);
			if (at.isNumeric())
				count++;
		}
		return (double)count;
	}

	/**
	 * Returns number of nominal attributes of the dataset (without class). 
	 * 
	 * @return number of nominal attributes of the dataset (without class).
	 */
	public double measureNumNominalAttributes() {
		int count = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			Attribute at = m_data.attribute(i);
			if (at.isNominal())
				count++;
		}
		return (double)count;
	}

	/**
	 * Returns whether there are missing values in the dataset or not (without class). 
	 *  
	 * @return 1.0 if there are missing values in the dataset or 0.0, if not.
	 */
	public double measureMissingValues() {
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			AttributeStats as = m_data.attributeStats(i);
			if (as.missingCount > 0)
				return (double)1.0;
		}
		return (double)0.0;
	}

	/**
	 * Returns number of attributes with missing values (without class). 
	 * 
	 * @return number of attributes with missing values (without class).
	 */
	public double measureNumAttsMissingValues() {
		int count = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			AttributeStats as = m_data.attributeStats(i);
			if (as.missingCount > 0)
				count++;
		}
		return (double)count;
	}

	/**
	 * Returns number of examples with missing values in the dataset (without class). 
	 * 
	 * @return number of examples with missing values in the dataset (without class).
	 */
	public double measureNumMissingValuesDataset() {

		int sum = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			AttributeStats as = m_data.attributeStats(i);
			sum += as.missingCount;
		}
		return (double)sum;
	}

	/**
	 * Returns percentage of examples with missing values in the dataset (without class). 
	 * 
	 * @return percentage of examples with missing values in the dataset (without class).
	 */
	public double measurePercentMissingValuesDataset() {

		int tamDataset = m_data.numInstances() * m_data.numAttributes();
		int sum = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			AttributeStats as = m_data.attributeStats(i);
			sum += as.missingCount;
		}
		return (double)sum/tamDataset*100.0;
	}

	/**
	 * Returns number of classes in the dataset. 
	 * 
	 * @return number of classes in the dataset.
	 */
	public double measureNumClasses() {

		return (double)m_data.numClasses();
	}

	/**
	 * Returns whether there are any empty classes or not. 
	 * 
	 * @return 1.0 if there are any empty classes or 0.0, if not. 
	 * 
	 */
	public double measureEmptyClass() {

		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int count = classCounts[Utils.minIndex(classCounts)];
		if (count == 0)
			return (double)1.0;
		else
			return (double)0.0;
	}

	/**
	 * Returns number of examples of the first class (considered by WEKA as positive by default)
	 * 
	 * @return number of examples of minority class.
	 */
	public double measureNumFirstClass() {

		int count=0;
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		count = classCounts[0];
		return (double)count;
	}

	/**
	 * Returns index of minority class (discarding empty classes). 
	 * 
	 * @return index of minority class.
	 */
	public double measureMinClassIndex() {
		
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int iClass, iMinClass;
		for(iClass = 0; ((iClass < classCounts.length) && (classCounts[iClass] == 0)); iClass++);
		iMinClass = iClass;
		for(iClass = iMinClass+1; (iClass < classCounts.length) ; iClass++) {
			if(classCounts[iClass] == 0)
				continue;
			if(classCounts[iClass] < classCounts[iMinClass])
				iMinClass = iClass;
		}
		return (double)iMinClass;
	}

	/**
	 * Returns number of examples of minority class
	 * (discarding empty classes). 
	 * 
	 * @return number of examples of minority class.
	 */
	public double measureNumMinClass() {

		int count=0;
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int i_iMinClass;
		for(i_iMinClass = 0; ((i_iMinClass < classCounts.length) && (Utils.kthSmallestValue(classCounts, i_iMinClass) == 0)); i_iMinClass++);
		if(i_iMinClass < classCounts.length)
			count = Utils.kthSmallestValue(classCounts,i_iMinClass);
		return (double)count;
	}

	/**
	 * Returns percentage of examples of minority class
	 * (discarding empty classes). 
	 * 
	 * @return percentage of examples of minority class.
	 */
	public double measurePercentMinClass() {

		int count=0;
		int numInstances = m_data.numInstances();
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int i_iMinClass;
		for(i_iMinClass = 0; ((i_iMinClass < classCounts.length) && (Utils.kthSmallestValue(classCounts, i_iMinClass) == 0)); i_iMinClass++);
		if(i_iMinClass < classCounts.length)
			count = Utils.kthSmallestValue(classCounts,i_iMinClass);
		return (double)count/numInstances*100.0;
	}

	/**
	 * Returns index of majority class (discarding empty classes). 
	 * 
	 * @return index of majority class.
	 */
	public double measureMajClassIndex() {
		
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int iClass, iMajClass;
		for(iClass = 0; ((iClass < classCounts.length) && (classCounts[iClass] == 0)); iClass++);
		iMajClass = iClass;
		for(iClass = iMajClass+1; (iClass < classCounts.length) ; iClass++) {
			if(classCounts[iClass] == 0)
				continue;
			if(classCounts[iClass] > classCounts[iMajClass])
				iMajClass = iClass;
		}
		return (double)iMajClass;
	}

	/**
	 * Returns number of examples of majority class. 
	 * 
	 * @return number of examples of majority class.
	 */
	public double measureNumMajClass() {

		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int count = classCounts[Utils.maxIndex(classCounts)];
		return (double)count;
	}

	/**
	 * Returns percentage of examples of majority class. 
	 * 
	 * @return percentage of examples of majority class.
	 */
	public double measurePercentMajClass() {

		int numInstances = m_data.numInstances();
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int count = classCounts[Utils.maxIndex(classCounts)];
		return (double)count/numInstances*100.0;
	}

	/**
	 * Returns an enumeration of the additional measure names
	 * 
	 * @return an enumeration of the measure names
	 */
	@Override
	public Enumeration<String> enumerateMeasures() {
		Vector<String> newVector = new Vector<String>(3);
		newVector.addElement("measureNumAttributes");
		newVector.addElement("measureNumNumericAttributes");
		newVector.addElement("measureNumNominalAttributes");
		newVector.addElement("measureMissingValues");
		newVector.addElement("measureNumAttsMissingValues");
		newVector.addElement("measureNumMissingValuesDataset");
		newVector.addElement("measurePercentMissingValuesDataset");
		newVector.addElement("measureNumClasses");
		newVector.addElement("measureEmptyClass");
		newVector.addElement("measureNumFirstClass");
		newVector.addElement("measureMinClassIndex");
		newVector.addElement("measureNumMinClass");
		newVector.addElement("measurePercentMinClass");
		newVector.addElement("measureMajClassIndex");
		newVector.addElement("measureNumMajClass");
		newVector.addElement("measurePercentMajClass");
		return newVector.elements();
	}

	/**
	 * Returns the value of the named measure
	 * 
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	@Override
	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.compareToIgnoreCase("measureNumAttributes") == 0) {
			return measureNumAttributes();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumNumericAttributes") == 0) {
			return measureNumNumericAttributes();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumNominalAttributes") == 0) {
			return measureNumNominalAttributes();
		} else if (additionalMeasureName.compareToIgnoreCase("measureMissingValues") == 0) {
			return measureMissingValues();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumAttsMissingValues") == 0) {
			return measureNumAttsMissingValues();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumMissingValuesDataset") == 0) {
			return measureNumMissingValuesDataset();
		} else if (additionalMeasureName.compareToIgnoreCase("measurePercentMissingValuesDataset") == 0) {
			return measurePercentMissingValuesDataset();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumClasses") == 0) {
			return measureNumClasses();
		} else if (additionalMeasureName.compareToIgnoreCase("measureEmptyClass") == 0) {
			return measureEmptyClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumFirstClass") == 0) {
			return measureNumFirstClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measureMinClassIndex") == 0) {
			return measureMinClassIndex();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumMinClass") == 0) {
			return measureNumMinClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measurePercentMinClass") == 0) {
			return measurePercentMinClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measureMajClassIndex") == 0) {
			return measureMajClassIndex();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumMajClass") == 0) {
			return measureNumMajClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measurePercentMajClass") == 0) {
			return measurePercentMajClass();
		} else {
			throw new IllegalArgumentException(additionalMeasureName
					+ " not supported (ZeroR)");
		}
	}

	/**
	 * Returns a table with the classes, the number of cases of each and the percentage they represent.
	 * 
	 * @return a summary of the the class distribution
	 */
	@Override
	public String toSummaryString() {

		int numInstances = m_data.numInstances();
		Attribute cl = m_data.classAttribute();
		//at.enumerateValues()
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;

		String lineResult = "Classes:\n";
		for(int i = 0; i < classCounts.length; i++) {
			if (i < classCounts.length - 1)
				lineResult = lineResult +  cl.value(i) +", ";
			else
				lineResult = lineResult + cl.value(i) + "\n";
		}
		for(int i = 0; i < classCounts.length; i++) {
			if (i < classCounts.length - 1)
				lineResult = lineResult +  classCounts[i] +", ";
			else
				lineResult = lineResult + classCounts[i] + "\n";
		}
		for(int i = 0; i < classCounts.length; i++) {
			if (i < classCounts.length - 1)
				lineResult = lineResult + Utils.doubleToString((double)classCounts[i]/numInstances*100, 3 + m_numDecimalPlaces, m_numDecimalPlaces) + ", ";
			else
				lineResult = lineResult + Utils.doubleToString((double)classCounts[i]/numInstances*100, 3 + m_numDecimalPlaces, m_numDecimalPlaces) + "\n";
		}
		return lineResult + " \n";
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier
	 */
	@Override
	public String toString() {
		return toSummaryString();
	}

}
