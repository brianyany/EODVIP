//Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/
//Partial reference is used to bridge Camera2BasicFragment.java; all referenced part in this file does not contain object detector's core algorithm

package com.example.yolodetector;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;
import android.text.style.RelativeSizeSpan;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;


public abstract class ImageDetector {
    // Display preferences
    private static final String TAG = "YoloDetector";

    private static final int RESULTS_TO_SHOW = 3;

    private static final int DIM_BATCH_SIZE = 1;

    private static final int PIXEL_DIM = 3;

    /** Preallocated buffers for storing image data in. */
    private int[] intValues = new int[getW() * getH()];

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** tflite model file */
    private MappedByteBuffer tfliteModel;

    /** tflite model interpreter */
    protected Interpreter tflite;

    /** Class names */
    private String[] labels;


    private static final float PROB_THRESHOLD = 0.3f;

    private static final int SECONDARY_COLOR = 0xffddaa88;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    protected ByteBuffer imgData = null;

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });


    ImageDetector(Activity activity) throws IOException {
        tfliteModel = loadModelFile(activity);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        labels = loadLabels();
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * getW() * getH() * PIXEL_DIM * getNumBytesPerChannel());
        imgData.order(ByteOrder.nativeOrder());
        Log.d(TAG, "Created a YOLOv2 Detector.");
    }

    /** Detects a frame from the camera, only called when the model is not running*/
    void detectFrame(Bitmap bitmap, SpannableStringBuilder builder) {
        if (tflite == null) {
//            Log.e(TAG, "Image detector has not been initialized; Skipped.");
            builder.append(new SpannableString("Uninitialized Detector."));
        }
        convertBitmapToByteBuffer(bitmap);
        long startTime = SystemClock.uptimeMillis();
        runInference();
        long cpTime = SystemClock.uptimeMillis();
//        Log.d(TAG, "Timecost to run model inference: " + Long.toString(cpTime - startTime));

        showResults();
        long endTime = SystemClock.uptimeMillis();
//        Log.d(TAG, "Timecost to run prediction: " + Long.toString(endTime - cpTime));
        // Print the results.
        printTopKLabels(builder);

        long duration1 = cpTime - startTime;
        long duration2 = endTime - cpTime;
        SpannableString span = new SpannableString("Inference: "+ duration1 + " ms Prediction: "+duration2+" ms");
        span.setSpan(new ForegroundColorSpan(android.graphics.Color.LTGRAY), 0, span.length(), 0);
        builder.append(span);
        sortedLabels.clear();
    }

    /**  Reference to tensorflow lite demo*/
    private void recreateInterpreter() {
        if (tflite != null) {
            tflite.close();
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }


    public void setNumThreads(int numThreads) {
        tfliteOptions.setNumThreads(numThreads);
        recreateInterpreter();
    }

    /** Closes tflite model, free memory
      Reference to tensorflow lite demo*/
    public void close() {
        tflite.close();
        tflite = null;
        tfliteModel = null;
    }


    /** load the .tflite file
     *  Reference to tensorflow lite demo*/
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }



    /** Writes Image data into a bytebuffer
     *  Reference to tensorflow lite demo*/
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
//        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < getW(); ++i) {
            for (int j = 0; j < getH(); ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
//        long endTime = SystemClock.uptimeMillis();
//        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    /** Prints top-K labels, to be shown in UI as the results.
     *Reference to tensorflow lite demo*/
    private void printTopKLabels(SpannableStringBuilder builder) {
        for (int i = 0; i < getNumLabels(); ++i) {
            sortedLabels.add(
//                    new AbstractMap.SimpleEntry<>(labelList.get(i), getProbability(i)));
            new AbstractMap.SimpleEntry<>(labels[i], getProbability(i)));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }

        final int size = sortedLabels.size();
        for (int i = 0; i < size; i++) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            SpannableString span =
                    new SpannableString(String.format("%s: %4.2f\n", label.getKey(), label.getValue()));
            int color;
            // Make it white when probability larger than threshold.
            if (label.getValue() > PROB_THRESHOLD) {
                color = android.graphics.Color.WHITE;
            } else {
                color = SECONDARY_COLOR;
            }
            // Make first item bigger.
            if (i == size - 1) {
                float sizeScale = (i == size - 1) ? 1.25f : 0.8f;
                span.setSpan(new RelativeSizeSpan(sizeScale), 0, span.length(), 0);
            }
            span.setSpan(new ForegroundColorSpan(color), 0, span.length(), 0);
            builder.insert(0, span);
        }
    }


    protected abstract String getModelPath();

    protected abstract int getW();

    protected abstract int getH();

    protected abstract int getNumBytesPerChannel();

    protected abstract float getProbability(int labelIndex);

    protected abstract void showResults();

    protected abstract String getObjLocation();

    /**
     * load label list from induvidual detector class
     */
    protected abstract String[] loadLabels();

    /**
     * Add pixelValue to byteBuffer.
     */
    protected abstract void addPixelValue(int pixelValue);

    /**
     * run inference on one image
     */
    protected abstract void runInference();

    /**
     * Get the total number of labels.
     */
    protected int getNumLabels() {
        return labels.length;
    }
}
