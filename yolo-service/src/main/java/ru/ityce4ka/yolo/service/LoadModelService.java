package ru.ityce4ka.yolo.service;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;

@Slf4j
@Service
public class LoadModelService {


    public ZooModel<Image, DetectedObjects> model;

    private LoadModelService() throws ModelNotFoundException, MalformedModelException, IOException {
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(416, 416));
        pipeline.add(new ToTensor());
        Translator<Image, DetectedObjects> translator = YoloV5Translator.builder().setPipeline(pipeline).optSynsetArtifactName("mnist.yaml").build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optDevice(Device.cpu())
                        .optModelUrls("yolo")
                        .optModelName("best.torchscript")
                        .optTranslator(translator)
                        .optArgument("truncation", 0.25f)
                        .optEngine("PyTorch")
                        .build();
        this.model = ModelZoo.loadModel(criteria);
    }
}
