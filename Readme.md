<h1 align="center"> Infralearning - engine </h1>

This program is part of the <a ref="https://github.com/victordalosto/infralearning">Infralearning</a> project, an AI program used to evaluate the assets of Road Network.

This program is the <b>engine</b> that uses the data of the <a href="https://github.com/victordalosto/infralearning-mounter">mounter</a> to create AI Models to be used by the <a href="https://github.com/victordalosto/infralearning-evaluate">evalutor</a> to predict and classify road assets. 
<br/><br/>


<h2> How it works </h2>

This engine uses <b>TensorFlow</b> to create AI models based on the data provided by the mounter. The model is saved in the <b>models</b> folder and can be used to predict and classify road assets.

The engine is abstract enough that it can be used to create AI models for any kind of data that uses images for classifications, not only road assets. 

The engine already creates out-of-the-box models for Binary Classification (0 or 1) (absent or present) and Multi Classification (0, 1, 2, 3, 4, 5, 6, 7, 8, 9).

