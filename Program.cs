using MongoDB.Bson;
using MongoDB.Driver;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;
using System;
using System.Linq;

var client = new MongoClient("mongodb://localhost:27017");
var collection = client.GetDatabase("tensorflow").GetCollection<BsonDocument>("sampleData");

var sampleData = new BsonDocument
{
    { "X", new BsonArray { 3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1 } },
    { "Y", new BsonArray { 1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3 } }
};

collection.InsertOne(sampleData);

Console.WriteLine("Data seeded successfully");

var document = collection.Find(new BsonDocument()).FirstOrDefault();

var xArray = document["X"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();
var yArray = document["Y"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();

var x = np.array(xArray);
var y = np.array(yArray);
var nSamples = x.shape[0];

var trainingSteps = 1000;
var learning_rate = 0.01f;
var displayStep = 100;

var w = tf.Variable(0.06f, name: "weight");
var b = tf.Variable(0.73f, name: "bias");
var optimizer = keras.optimizers.SGD(learning_rate);

foreach(var step in range(1, trainingSteps + 1))
{
    using var g = tf.GradientTape();
    var pred = w * x + b;
    var loss = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * nSamples);
    var gradients = g.gradient(loss, (w, b));
   optimizer.apply_gradients(zip(gradients, (w, b)));

    if (step % displayStep == 0)
    {
        pred = w * x + b;
        loss = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * nSamples);
        Console.WriteLine($"step: {step}, loss: {loss.numpy()}, w: {w.numpy()}, b: {b.numpy()}");
    }
}