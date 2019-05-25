package com.example.yolodetector;

import android.graphics.RectF;
import java.util.List;

public interface Detector {

    public class Recognition {
        /**Class id for an detected object*/
        private final int id;

        /**Class name for an detected object*/
        private final String name;

        /**Confidence for an detected object*/
        private final Float confidence;

        /**Location of an detected object*/
        private RectF location;

        public Recognition(
                final int id, final String name, final Float confidence, final RectF location) {
            this.id = id;
            this.name = name;
            this.confidence = confidence;
            this.location = location;
        }

        public int getId() {
            return id;
        }

        public String getName() {
            return name;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

    }

    /**
     * Template function to recognize an image
     */
    List<Recognition> recognizeImage();
}