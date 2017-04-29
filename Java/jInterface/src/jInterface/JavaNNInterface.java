package jInterface;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.ArrayUtils;

public class JavaNNInterface {
	
	public native void test(String in);
	
	public native void executeNeuralNet(int[] nodes_per_layer, int layers,
										int batch_size, float alpha, float[] inputs, 
										float[] act, int train_rows, int train_cols, 
										String save_path  );
	private float[] readCSV (String path){
		return ArrayUtils.toPrimitive(readCSV(path, true));
	}
	
	private Float[] readCSV (String path, boolean go){
		try {
			Reader in = new FileReader(path);
			Iterable<CSVRecord> records = CSVFormat.EXCEL.parse(in);
			ArrayList<Float> data_list= new ArrayList<Float>();
			for (CSVRecord record : records){
				String rec = record.get(0);
				
				if (!rec.isEmpty()){
					data_list.add(Float.parseFloat(rec));
				} 
				
			}
			return data_list.toArray(new Float[data_list.size()]);
		} catch (FileNotFoundException ex) {
			System.out.println("Java could not find file while reading CSV");
			ex.printStackTrace();
		} catch (IOException io) {
			// TODO Auto-generated catch block
			System.out.println("Java IOException while reading CSV");
			io.printStackTrace();
		}
		return null;
	}
	
	static { 
		System.loadLibrary("neuralnet");
		}
	
	public static void main (String[] args){
		JavaNNInterface jnni = new JavaNNInterface();
		jnni.test("hi");
		float[] inputs = jnni.readCSV("C:/Users/albgk/Documents/GitHub/ML-NN-Repo/testCases/inputs_short.csv");
		float[] act = jnni.readCSV("C:/Users/albgk/Documents/GitHub/ML-NN-Repo/testCases/actual_short.csv");
		String save_path = "C:/Users/albgk/Documents/GitHub/ML-NN-Repo/testCases/output/out_java.csv";
		int[] nodes_per_layer = {1, 24, 1};
		jnni.executeNeuralNet(nodes_per_layer, nodes_per_layer.length, 5, (float)0.8, inputs, act, inputs.length, 1, save_path);
		System.out.flush();
		System.out.println("exited successfully");
	}
}

