# import important libraries 
from flask import Flask, render_template,request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Load the saved tf keras model 
model = tf.keras.models.load_model(
    'my_trained_model.h5') 
# Below is the name for each classes 
class_labels = {
    0: 'Pepper_bell_Bacterial_spot',
    1: 'Pepper_bell_healthy',
    2: 'Potato_Early_blight',
    3: 'Potato_healthy',
    4: 'Potato_Late_blight',
    5: 'Tomato_Target_Spot',
    6: 'Tomato_Tomato_mosaic_virus',
    7: 'Tomato_Tomato_YellowLeaf_Curl_Virus',
    8: 'Tomato_Bacterial_spot',
    9: 'Tomato_Early_blight',
    10: 'Tomato_healthy',
    11: 'Tomato_Late_blight',
    12: 'Tomato_Leaf_Mold',
    13: 'Tomato_Septoria_leaf_spot',
    14: 'Tomato_Spider_mites_Two_spotted_spider_mite'
}
# Below is the result_text for each class that will be passed to the result html page below
result_text={
    0: 'Pepper bacterial spot is a plant disease caused by the bacterium Xanthomonas campestris pv. vesicatoria. It primarily affects pepper plants, causing small, dark, water-soaked lesions on the leaves with yellow or greenish halos. Infected fruit may develop raised, dark spots with sunken centers. This disease can lead to reduced yields and deformed fruit. It spreads through water, survives on plant debris and seeds, and is exacerbated by rainy or humid conditions. Management involves using resistant varieties, crop rotation, sanitation, drip irrigation, copper-based sprays, biological control, and proper spacing to prevent and control the disease',
    1: 'Healthy bell peppers, also known as sweet peppers or capsicums, are nutritious vegetables that are rich in essential nutrients, such as vitamin C, vitamin A, and dietary fiber. They can be part of a balanced diet when you choose a variety of colors, opt for organic or pesticide-free options, and enjoy them fresh and raw in salads or as a crunchy snack. When cooking bell peppers, use healthy methods like grilling or roasting and add herbs and spices for flavor. Bell peppers are naturally low in calories and can be used in a wide range of dishes, contributing to a diverse and nutritious diet',
    2: 'Potato early blight is a fungal disease caused by Alternaria solani that affects potato plants. It can lead to reduced yields and lower-quality potatoes. Control measures include the use of fungicides like chlorothalonil, mancozeb, and copper-based products. Additionally, cultural practices such as crop rotation, planting resistant potato varieties, proper spacing, and good sanitation can help manage the disease. It is important to follow label instructions and consider integrated pest management techniques for effective control while minimizing environmental impact.',
    3: 'Healthy potatoes are potatoes that are grown, prepared, and consumed in a way that maximizes their nutritional value and minimizes potential health risks. This includes selecting varieties with lower starch and higher nutrients, choosing organic or pesticide-free options, using healthier cooking methods like boiling or steaming, leaving the skin on for added fiber and nutrients, and practicing portion control. Healthy potatoes can be part of a balanced diet when prepared sensibly and combined with a variety of other foods.',
    4: '-Potato late blight, caused by Phytophthora infestans, is a devastating fungal disease. To manage it, fungicides like chlorothalonil, metalaxyl-M, and mancozeb can be used. Cultural practices such as crop rotation, planting resistant potato varieties, proper spacing, sanitation, and early detection are key for control. Weather monitoring is important, as late blight thrives in cool, wet conditions. Integrated pest management (IPM) is recommended to effectively manage the disease while minimizing environmental impact. Rapid action is crucial to minimize the impact of this destructive disease.',
    5: 'Tomato Target Spot is a foliar disease that affects tomato plants. It is characterized by the development of small, dark brown to black spots on the leaves, often with concentric rings or a target-like appearance. To manage it, practice good sanitation by removing infected plant material, enhance airflow through pruning and proper spacing, manage water to keep foliage dry, use organic mulch, and consider fungicides with active ingredients like chlorothalonil or mancozeb for severe cases. Planting resistant tomato varieties can also be effective. Integrated pest management (IPM) techniques are recommended, especially during periods of high humidity and wet conditions, to effectively control Tomato Target Spot',
    6: 'Tomato Mosaic Virus (ToMV) is a viral disease that affects tomato plants, causing mosaic-like patterns of light and dark green on the leaves, stunted growth, and reduced fruit quality and yield. Managing ToMV involves preventing its introduction through disease-free seeds and transplants, isolating infected plants, practicing good hygiene, controlling aphids and whiteflies (which transmit the virus), using reflective mulch, planting resistant varieties, pruning and removing infected plant parts, and, in some cases, using fungicides to limit virus spread. Prevention is key, as there are no cures for infected plants. Consult with local experts for region-specific guidance on managing ToMV.',
    7: 'Tomato Yellow Leaf Curl Virus (TYLCV) is a viral disease that affects tomato plants and is transmitted by whiteflies. It causes symptoms such as leaf curling, yellowing, and stunted growth. To manage TYLCV, control measures include whitefly management, using resistant tomato varieties, employing reflective mulch and physical barriers, pruning and removing infected plants, and practicing good sanitation. Early detection and monitoring are crucial for effective management, as there is no cure once a plant is infected. Consult with local agricultural experts for region-specific guidance on managing TYLCV.',
    8: 'Tomato Bacterial Spot is a common bacterial disease that affects tomato plants, causing small, water-soaked lesions on the leaves, stems, and fruit, which may turn dark and necrotic. To manage it, practice sanitation by removing infected plant material, improve air circulation through pruning and proper spacing, manage water to keep the foliage dry, and use copperbased products. Planting resistant tomato varieties and monitoring weather conditions are also important. Integrated pest management (IPM) techniques are recommended to effectively control Tomato Bacterial Spot.',
    9: 'Tomato early blight is a common fungal disease that affects tomato plants, characterized by dark concentric rings on the leaves that eventually turn yellow and brown. To manage it, practice crop rotation, sanitation, pruning, proper spacing, and watering techniques that keep foliage dry. Fungicides containing chlorothalonil can be used for control. Some tomato varieties are resistant to early blight. Early detection and integrated pest management (IPM) techniques are essential for effective control, particularly in periods of high humidity and wet conditions.',
    10: 'Tomatoes are nutrient-rich vegetables known for their numerous health benefits. They are a good source of essential vitamins, such as vitamin C and K, and are rich in antioxidants like lycopene, which can reduce the risk of chronic diseases and certain cancers. Tomatoes promote heart health, skin health, and eye health, help with digestion, and aid in weight management. Their versatility in cooking makes them a popular and delicious addition to various dishes. Additionally, tomatoes are low in calories, sodium, and cholesterol, making them a nutritious choice for a well-balanced diet.',
    11: 'Tomato late blight is a severe and destructive disease caused by the Phytophthora infestans pathogen, famous for causing the Irish Potato Famine. It can rapidly defoliate tomato plants and cause fruit rot. To manage it, fungicides are often necessary, including chlorothalonil, metalaxyl-M (mefenoxam), and phosphorous acid-based products. Preventive spraying may be required. Other strategies include proper spacing, pruning, water management, organic mulch, planting resistant varieties, and early detection. Late blight requires vigilant management, especially during periods of high humidity and wet conditions. Integrated pest management techniques are recommended to minimize the environmental impact while controlling the disease',
    12: 'Tomato leaf mold is a fungal disease that affects tomato plants, characterized by yellowing, pale green or brown patches on the leaves with fuzzy gray to brown growth on the undersides. To manage it, practice proper plant spacing, prune and remove lower leaves, manage water to keep foliage dry, use organic mulch, and consider fungicides containing copper or chlorothalonil. Some tomato varieties are resistant to leaf mold. Early detection and integrated pest management (IPM) are crucial for effective control, particularly during periods of high humidity.',
    13: 'Septoria leaf spot is a common fungal disease that affects tomato plants, characterized by small, circular lesions with gray centers and dark margins on the leaves. To manage it, employ practices such as crop rotation, sanitation, pruning, proper spacing, mulching, and careful watering. Fungicides can be used if the disease is severe, and resistant tomato varieties are available. Early detection and integrated pest management (IPM) techniques are crucial for effective control while minimizing environmental impact',
    14: 'Two-spotted spider mites are common pests that can infest tomato plants. They damage the plants by feeding on their sap, leading to stippling and discoloration of the leaves. To manage these pests, consider practices like maintaining proper soil moisture, using water sprays to dislodge mites, introducing beneficial predators, and using remedies such as neem oil or insecticidal soap. Miticides can be used for severe infestations, but non-chemical control methods should be the first line of defense. Regular monitoring and early intervention are key to preventing severe infestations.'
}
# Below is the links for each class that will be passed to the result html page below
links={
    0: 'https://marronebio.com/buy/regalia-cg/',
    1: 'https://www.amazon.in/Grixisonsazia-Capsicum-Vegetable-Gardening-Planting/dp/B09ZRTVN6P/ref=sr_1_2?crid=3UVRI5QXD80S5&keywords=pepper+bell&qid=1698007687&sprefix=pepper+bell%2Caps%2C223&sr=8-2',
    2: 'https://agribegri.com/products/buy-indofil-z-78-zineb-75-wp-fungicides-agribegri-online-agro-store.php',
    3: 'https://www.bigbasket.com/pd/10000159/fresho-potato-1-kg/',
    4: 'https://agribegri.com/products/buy-indofil-z-78-zineb-75-wp-fungicides-agribegri-online-agro-store.php',
    5: 'https://www.amazon.in/Bayer-Premise-250ml-Construction-Contruction/dp/B071XXKFQV/ref=asc_df_B071XXKFQV/?tag=googleshopdes-21&linkCode=df0&hvadid=396984869172&hvpos=&hvnetw=g&hvrand=14868654921261890695&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9301936&hvtargid=pla-837304019920&psc=1&ext_vrnc=hi',
    6: 'https://www.bighaat.com/products/ridomill-gold-fungicide',
    7: 'https://farmmate.in/products/pyriproxyfen-10-ec-insecticide?variant=40329586278571&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&gclid=Cj0KCQjw7JOpBhCfARIsAL3bobe_566A1TAmbEnbU3O66UQtkekd8u6aR93Z6AKQOnYgenddOizoPLkaAgOpEALw_wcB',
    8: 'https://farmkey.in/product/kocide-250-gm?gclid=Cj0KCQjw7JOpBhCfARIsAL3bobcVPqrdS656UKvQaoR4h5CDTDPvNlb0c4PDxef89iSufp77IDJCoQgaAnXUEALw_wcB',
    9: 'https://agribegri.com/products/buy-adrone-azoxystrobin-182--difenoconazole-114-sc.php',
    10: 'https://www.bigbasket.com/pd/10000203/fresho-tomato-local-1-kg/?nc=cl-prod-list&t_pos_sec=1&t_pos_item=1&t_s=Tomato+-+Local+%2528Loose%2529',
    11: 'https://www.bighaat.com/products/ridomill-gold-fungicide',
    12: 'https://www.ubuy.co.in/product/BW3NTYG-daconil-fungicide-concentrate-16-oz-100523634',
    13: 'https://www.amazon.in/AD-45-Mancozeb-75-WP-Fungicide/dp/B07JDSJY6C',
    14: 'https://agribegri.com/products/safex-permethrin-25-ec-pixel-25.php'
}


# Function to predict the disease and then return predicted_class , result_text , result_link to render on result.html 
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    class_idx = np.argmax(predictions)
    predicted_class_label = class_labels[class_idx]
    result_text1=result_text[class_idx]
    result_link=links[class_idx]

    return predicted_class_label,result_text1,result_link

# Create the main app using Flask 
app=Flask(__name__)

@app.route('/',methods=['GET'])
def func():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def pred():
    # take image from the form and save it to the server in static/images folder
    imagefile=request.files['imagefile']
    image_path="./static/images/"+imagefile.filename
    imagefile.save(image_path)

    # predict result from above function 
    prediction,result_text1,result_link1 = predict_image_class(image_path)
    
    # render template with variables 
    return render_template('result.html', prediction=prediction, result_text1=result_text1,result_link1=result_link1, new_image_path=image_path)


    
if __name__=='__main__':
    app.run(host='0.0.0.0' ,debug=True)
