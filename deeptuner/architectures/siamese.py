from tensorflow.keras import layers, Model, Input

class SiameseArchitecture:
    def __init__(self, input_shape, embedding_model):
        self.input_shape = input_shape
        self.embedding_model = embedding_model

    def create_siamese_network(self):
        anchor_input = Input(name="anchor", shape=self.input_shape)
        positive_input = Input(name="positive", shape=self.input_shape)
        negative_input = Input(name="negative", shape=self.input_shape)

        anchor_embedding = self.embedding_model(anchor_input)
        positive_embedding = self.embedding_model(positive_input)
        negative_embedding = self.embedding_model(negative_input)

        outputs = [anchor_embedding, positive_embedding, negative_embedding]
        model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=outputs, name="siamese_network")
        return model
