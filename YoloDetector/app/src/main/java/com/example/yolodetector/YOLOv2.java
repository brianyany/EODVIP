package com.example.yolodetector;

import android.app.Activity;
import android.graphics.RectF;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;

public class YOLOv2 extends ImageDetector implements Detector {


    private float[] labelProbArray = null;
    private float[][][][] pred = null;

    private String[] loc = null;
    private boolean locFlag = false;
    private boolean locFlagR = true;

    private static final int MAX_RESULTS = 5;

    private static final int NUM_CLASSES = 80;

    private static final int NUM_BOXES_PER_BLOCK = 5;

    private static final int GRID_WIDTH = 13;
    private static final int GRID_HEIGHT = 13;

    private static final int BLOCK_SIZE = 32;

    private int actual_size = 0;

    private static final double[] ANCHORS = {
            0.57273, 0.677385,
            1.87446, 2.06253,
            3.33843, 5.47434,
            7.88282, 3.52778,
            9.77052, 9.16828
    };

    private static final String[] LABELS = {
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "television",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
    };

    YOLOv2(Activity activity) throws IOException {
        super(activity);
        labelProbArray = new float[getNumLabels()];
        pred = new float[1][13][13][425];
        loc = new String[getNumLabels()];
    }

    @Override
    protected String[] loadLabels(){
        return LABELS;
    }

    @Override
    protected String getModelPath() {
        return "tiny_model.tflite";
    }

    @Override
    protected int getW() {
        return 416;
    }

    @Override
    protected int getH() {
        return 416;
    }

    @Override
    protected int getNumBytesPerChannel() {
        return 4; // Float.SIZE / Byte.SIZE;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.f);
        imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.f);
        imgData.putFloat((pixelValue & 0xFF) / 255.f);
    }

    @Override
    protected float getProbability(int labelIndex) {
        return labelProbArray[labelIndex];
    }

    @Override
    protected void runInference() {
        tflite.run(imgData, pred);
    }


    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    private float iou(RectF r1, RectF r2){
        float b = Math.max(r1.bottom, r2.bottom);
        float l = Math.max(r1.left, r2.left);
        float r = Math.max(r1.right, r2.right);
        float t = Math.max(r1.top, r2.top);

        float intersection = Math.max(0,b-t+1)*Math.max(0,r-l+1);
        float area1 = (r1.bottom-r1.top+1)*(r1.right-r1.left+1);
        float area2 = (r2.bottom-r2.top+1)*(r2.right-r2.left+1);

        return intersection / (area1+area2-intersection);
    }

    @Override
    public ArrayList<Recognition> recognizeImage(){
        final PriorityQueue<Detector.Recognition> pq =
                new PriorityQueue<Detector.Recognition>(1,
                        new Comparator<Detector.Recognition>() {
                            @Override
                            public int compare(final Detector.Recognition R1, final Detector.Recognition R2) {
                                return Float.compare(R2.getConfidence(), R1.getConfidence());
                            }
                        });
//        Set<Classifier.Recognition> rset = new HashSet<Classifier.Recognition>();
//        float[][] classes = new float[GRID_WIDTH*GRID_HEIGHT*NUM_BOXES_PER_BLOCK][NUM_CLASSES];
        for (int y = 0; y < GRID_HEIGHT; ++y) {
            for (int x = 0; x < GRID_WIDTH; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {

                    final float xPos = (x + expit(pred[0][y][x][b*85])) * BLOCK_SIZE;
                    final float yPos = (y + expit(pred[0][y][x][b*85+1])) * BLOCK_SIZE;

                    final float w = (float) (Math.exp(pred[0][y][x][b*85+2]) * ANCHORS[2 * b + 0]) * BLOCK_SIZE;
                    final float h = (float) (Math.exp(pred[0][y][x][b*85+3]) * ANCHORS[2 * b + 1]) * BLOCK_SIZE;

                    final RectF rect = new RectF(
                                    Math.max(0, xPos - w / 2),
                                    Math.max(0, yPos - h / 2),
                                    Math.min(416 - 1, xPos + w / 2),
                                    Math.min(416 - 1, yPos + h / 2));

                    final float confidence = expit(pred[0][y][x][b*85+4]);

                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = pred[0][y][x][b*85+5+c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }
                    final float confidenceInClass = maxClass * confidence;
                    if (confidenceInClass > 0.01) {
                        pq.add(new Detector.Recognition(detectedClass, LABELS[detectedClass], confidenceInClass, rect));
                    }
                }
            }
        }
        ArrayList<ArrayList<Detector.Recognition>> cla = new ArrayList<ArrayList<Detector.Recognition>>(80);
        //initialize the array
        for (int i=0; i<NUM_CLASSES; i++){
            ArrayList<Detector.Recognition> temparray  = new ArrayList<Detector.Recognition>();
            RectF temploc = new RectF(0,0,0,0);
            float tempconf = 0.0f;
            temparray.add(new Detector.Recognition(-1, LABELS[0], tempconf, temploc));
            cla.add(temparray);
        }
        //fill in the array with objects
        for (int i = 0; i <pq.size(); ++i){
            Detector.Recognition temp = pq.poll();
            ArrayList<Detector.Recognition> temparray = cla.get(temp.getId());
            temparray.add(temp);
            if (temparray.get(0).getId()==-1){
                temparray.remove(0);
            }
            cla.set(temp.getId(),temparray);
        }



        for (int i=0; i<cla.size(); i++){
            ArrayList<Detector.Recognition> temparray = cla.get(i);
            if (temparray.get(0).getId()!=-1){
                while (temparray.size()!=0){
                    Detector.Recognition temp = temparray.get(0);
                    int j=1;

                    while (j<temparray.size()){
                        if (iou(temp.getLocation(), temparray.get(j).getLocation())>0.5){
                            temparray.remove(j);
                        }
                        else{
                            j+=1;
                        }
                    }
                    pq.add(temparray.remove(0));
                }
            }

        }

        final ArrayList<Detector.Recognition> recognitions = new ArrayList<Detector.Recognition>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;     //Recogn -> location -> 4 values
    }

    @Override
    protected void showResults(){
        ArrayList<Detector.Recognition> recognitions  = recognizeImage();
        labelProbArray = new float[getNumLabels()];
        if (locFlagR){
            locFlag = false;
        }
        if(!locFlag) {
            actual_size = 0;
            for (int i = 0; i < recognitions.size(); i++) {
                labelProbArray[recognitions.get(i).getId()] = recognitions.get(i).getConfidence();
                if (recognitions.get(i).getConfidence()>=0.3) {
                    RectF pos = recognitions.get(i).getLocation();
                    int t = (int) (pos.top + pos.bottom) / (2*138);
                    int l = (int) (pos.left + pos.right) / (2*138);
                    String region = Integer.toString(3 * t + l + 1);
                    loc[i] = recognitions.get(i).getName() + " " + region;
                    actual_size+=1;
                }
            }
        }
    }

    @Override
    protected String getObjLocation(){
        locFlag = true;
        locFlagR = false;
        String ret = "";
        if (loc != null){
            for (int i=0; i<Math.min(actual_size, MAX_RESULTS); i++){
                ret = ret+loc[i]+"    ";
            }
            locFlagR = true;
            return ret;
        }
        else{
            locFlagR = true;
            return "Empty Array";
        }
    }

}
