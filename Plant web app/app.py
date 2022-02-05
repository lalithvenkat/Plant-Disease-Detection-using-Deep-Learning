import os
import numpy as np
import io
from flask import Flask, render_template, url_for, request, send_file
from werkzeug.utils import secure_filename
import cv2

from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from io import BytesIO
import base64
from tensorflow.keras.models import load_model
import pickle


from skimage.transform import resize

model = load_model("VGG16(m).h5")

app = Flask(__name__)

picf=os.path.join('static','images')

app.config['upf']=picf


with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/doc',methods=['GET', 'POST'])
def docx():
    name=['Apple Scab','Apple Black rot','Apple cedar apple rust','Cherry Powdery mildew','Corn cercospora leaf spot','Corn common rust','Corn Northen leaf blight','Grape black rot','Grape Black measles','Grape leaf Isariopsis leaf spot','orange huanglongbing(citrus greening) disease','Peach leaf bacterial spot','Pepper bell bacterial spot','Potato leaf  early blight','Potato leaf late blight','Squah Powdey mildew','Strawberry leaf scorch','Tomato Bacterial spot','Tomato early blight','Tomato late blight','Tomato leaf mold','Tomato Septoria leaf spot','Tomato  Two spotted spider mite disease','Tomato Target spot','Tomato mosaic virus','Tomato yellow leaf curl virus','Apple Healthy leaves','Blueberry healthy','Corn healthy','Grape Healthy','Peach healthy','Pepper bell Healthy','Potato Healthy','Raspberry Healthy','Soyabean Healthy','Strawberry Healthy','Cherry healthy','Tomato healthy']
    data=""
    control=""
    content=""
    pic=""
    if request.method=='POST':

        dname=request.form.get('selectop')

        if dname=='Apple Scab':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            #print(pic)
            content="The fungus Venturia inaequalis causes apple scab, which spreads by airborne spores and survives the winter on fallen leaves. Scab marks will form on leaves from mid-spring through autumn leaf fall"
            control="First line of defense should be good cultivation methods, cultivar selection, and garden hygiene and Keeping pests at bay and introducing natural enemies The.If chemical controls are utilized, they should be applied sparingly and with great precision.Fungicides such as Tebuconazole,Myclobutanil, Captan, Chlorothalonil, Propiconazole should be used"
        elif dname=='Apple Black rot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Black rot is an apple disease caused by the fungus Botryosphaeria obtusa, which attacks the fruit, leaves, and bark. Fruit infection is the most harmful type of this virus, and it starts with infected blooms before spreading to the fruits."
            control="Remove contaminated plant material from the region and prune out dead or diseased branches. Chemicals like Captan, as well as sulfur products, are employed."
        elif dname=='Apple cedar apple rust':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The fungus Gymnosporangium juniper-virginianae causes cedar apple rust. It's sometimes confused with other rust diseases, but they're not the same.In the spring, it infects apples and crabapples, and later in the summer, it attacks juniper bushes. The fungus is far more harmful to apple trees than it is to juniper trees"
            control="Using a combination of cultural approaches and chemical treatments, the best strategy to control cedar apple rust is to prevent infection.Pruning the branches approximately 4-6 inches below the galls to cut juniper galls Dip your pruning shears in 10% bleach or alcohol for at least 30 seconds between cuts to disinfect them.Fungicides such as sterol inhibitors and Immunox should be utilized"
        elif dname=='Cherry Powdery mildew':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Powdery mildew is caused by the obligate biotrophic fungus Podosphaera clandestina. Sweet cherry (Prunus avium) cultivars in the middle and late seasons are usually damaged, rendering them unmarketable due to a covering of white fungal growth on the fruit."
            control="Avoid crowding plants to provide proper air circulation.Don't fertilize excessively.Chemicals such as sodium bicarbonate, potassium bicarbonate, sulfur, and lime/sulfur, as well as Vinegar can be used."
        elif dname=='Corn cercospora leaf spot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Grey leaf spot (GLS) is a foliar fungus that affects maize (corn). GLS is often regarded as one of the most important yield-restricting diseases in maize.  Cercospora zeae-maydis and Cercospora zeina are the two fungal infections that cause GLS."
            control="Avoid overwatering or watering in the late evening to reduce free moisture.Space plants to encourage air movement and reduce high humidity levels.Use fungicides like Headline EC, Quilt, Proline 480 SC, Tilt 250 E, Bumper 418 EC are used."
        elif dname=='Corn common rust':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The fungus Puccinia sorghi causes common corn rust, which is the most common of the two principal rust diseases of corn in the United States.Rust-colored to dark brown, elongated pustules appear on both leaf surfaces with common rust."
            control="A foliar fungicide is used to treat the leaves of the plant.Plants that are genetically resistant should be chosen Aproach®, Headline®, Headline SC, Headline AMP®, PropiMax® EC, Quadris®, Quilt®, Quilt Xcel®, Stratego®, Stratego® YLD, and Tilt® are some of the most widely used fungicides."
        elif dname=='Corn Northen leaf blight':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Exserohilum turcicum, an anamorph of the ascomycete Setosphaeria turcica, causes Northern corn leaf blight (NCLB) or Turcicum leaf blight (TLB), a foliar disease of corn (maize). This disease can cause significant yield loss in vulnerable corn hybrids due to its unique cigar-shaped lesions."
            control="Apply fungicides on corn at the R1 (early silking) stage, Using Resistant hybrids, and Delaro fungicide gives preventative and curative defense."
        elif dname=='Grape black rot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The fungus Guignardia bidwellii causes black rot, which is a devastating disease of both cultivated and wild grapes. In warm, wet seasons, the illness is most destructive. All green portions of the vine are attacked, including leaves, shoots, leaf and fruit stems, tendrils, and fruit. The fruit is the one who suffers the most."
            control="Plantations with adequate spacing, sunlight, and air circulation In the early winter, when the vines are dormant, the vines need to be pruned. Copper, captan, ferbam, mancozeb, maneb, triadimefon, and ziram are some of the protective fungicides that can be used."
        elif dname=='Grape Black measles':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Grapevine measles, also known as esca, black measles, or Spanish measles, has long been a source of consternation for grape producers due to its cryptic symptoms and, for a long time, a lack of an identifiable causative organism (s).The symptoms of black measles can be seen on the leaves, which have a tiger-stripe pattern on them."
            control="To minimize inoculum levels, dormant sprays are used.Scout early and scout frequently.Fungicides such as mancozeb and ziram are used as preventative and systemic fungicides."
        elif dname=='Grape leaf Isariopsis leaf spot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Isariopsis is a fungus genus belonging to the Mycosphaerellaceae family. Pseudocercospora vitis, formerly known as I. vitis, causes the plant disease known as isariopsis leaf spot."
            control="Planting varieties that are less sensitive; avoiding excessive late-season fertilizers.Pruning the canopy cover to enable for proper foilage aeration While the grapes are dormant, a Bordeaux mixture or other appropriate fungicide may be required."
        elif dname=='orange huanglongbing(citrus greening) disease':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Huanglongbing (HLB), often known as citrus greening, is the most serious citrus disease now wreaking havoc on the global citrus industry. Candidatus Liberibacter spp., the likely pathogenic bacterial agent, affects tree health as well as fruit development, ripening, and quality of citrus fruits and juice."
            control="There is no cure for citrus greening; the only approach to safeguard trees is to stop the HLB disease from spreading by suppressing psyllid populations and killing any afflicted trees.Bactericides are a topical therapy for citrus greening that works by delaying the bacteria that causes it."
        elif dname=='Peach leaf bacterial spot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="This disease affects peach fruit and leaves and is caused by the bacteria Xanthomonas arboricola pv. pruni. Small reddish-purple dots form on infected leaves, with a white center in many cases. The inner portion of the spot often falls out in advanced cases, leaving the leaf looking 'ragged' or 'shot-holed'. Leaves that have become infected turn yellow and fall off the tree."
            control="Spraying plant activators.Biological or microbiological products are used..Streptomycin is used to treat transplant recipients.Spraying fungicides containing copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan in high concentrations."
        elif dname=='Pepper bell bacterial spot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Bacterial leaf spot is the most prevalent and destructive disease of peppers in the eastern United States, caused by Xanthomonas campestris pv. vesicatoria. It's a gram-negative, rod-shaped bacterium that may persist from one season to the next in seeds and plant waste."
            control="Seed treatment with hot water, which involves soaking seeds for 30 minutes in water that has been preheated to 125 F/51 C, is successful in lowering bacterial populations.Spraying early and often is the key.Control of bacterial leaf spot with copper and Mancozeb sprays.Quintec (Dow AgroSciences) and Actigard (Syngenta Crop Protection) are also effective preventatives."
        elif dname=='Potato leaf  early blight':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Phytophthera Infestans is an oomycete pathogen that causes potato blight. Blighted tubers help the fungus survive the winter. These could be left in the soil after the previous crop or in dumps where potatoes were dumped after grading. The disease could potentially be spread by planting blighted tubers."
            control="Bayer Garden Blight Control, which can be used up to four times every growing season, is sprayed on the plants.Planting a blight-resistant type of potato is the best approach to avoid blight.removing diseased stems and foliageTo prevent spores from entering into the potatoes, make sure they are well earthed."
        elif dname=='Potato leaf late blight':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Potato (late blight) is a disease produced by a fungus-like organism that causes collapse and degradation in the leaves, tubers, and fruit of potatoes in rainy weather. It is a severe disease that affects potatoes."
            control="After harvest, all potato trash is destroyed.After heavy rain or when the amount of disease is rapidly increasing, apply a copper-based fungicide (2 oz/gallon of water) every 7 days or fewer.Organocide® Plant Doctor, used as a foliar spray, will work its way through the entire plant to avoid fungal diseases.SERENADE Garden is a safe way to treat fungal diseases."
        elif dname=='Squah Powdey mildew':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Powdery mildew infects all cucurbits, including muskmelons, squash, cucumbers, gourds, watermelons, and pumpkins, and is caused primarily by the fungus Podosphaera xanthii. Powdery mildew can kill leaves prematurely, reducing output and quality of fruit in extreme cases."
            control="Mix one gallon of water with one tablespoon baking soda and one-half teaspoon liquid, non-detergent soap, then spray the mixture liberally on the plants.Mouthwash, which we use on a regular basis to eliminate bacteria in our mouths, can also be used to kill powdery mildew spores.Chlorothalonil, copper, and sulpur are the most often used fungicides."
        elif dname=='Strawberry leaf scorch':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The fungus Diplocarpon earliana causes leaf scorch. Leaf scorch manifests itself as a smattering of small, irregular reddish spots or 'blotches' on the upper surface of leaves. The blotches' centers turn a brownish color."
            control="The strawberry patch's diseased garden detritus was removed.Maintaining a steady strawberry production requires the planting of fresh plantings and strawberry patches.Avoiding soggy soil and cleaning up your garden on a regular basis will assist to prevent the spread of this fungus."
        elif dname=='Tomato Bacterial spot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Tomato bacterial spot is a potentially disastrous disease that can result in unmarketable fruit and even plant death in severe situations. Bacterial spot can appear anywhere tomatoes are cultivated, but it is more common in hot, humid areas in greenhouses."
            control="A plant with bacterial spot cannot be curedTo avoid the spread of germs to healthy plants, remove symptomatic plants from the field or greenhouse.At the conclusion of the season, burn, bury, or compost tomato trash in hot compost.To avoid the spread of bacterial spot infections, only use pathogen-free seed or transplants."
        elif dname=='Tomato early blight':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The fungus Alternaria solani causes early blight, which is a common tomato disease. It can harm the leaves, stems, and fruits of tomato plants, among other things. Although the plants are unlikely to die, they will be weakened and produce fewer tomatoes than usual."
            control="Crop rotation is practiced by planting tomatoes in an area of the garden that has never been used before.Mancozeb and Zoxamide, Chlorothalonil, Cyprodinil, and Fludioxonil are some of the most often used fungicides for the control of tomato early blight."
        elif dname=='Tomato late blight':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Late blight is a potentially fatal tomato and potato disease that affects the leaves, stems, and fruits of tomato plants. The disease spreads swiftly in fields and, if left untreated, can lead to catastrophic crop collapse. The Irish potato famine of the late 1840s was caused by late blight of the potato."
            control="Clean up the garden area of all rubbish and fallen fruit.Late blight is a fatal disease that cannot be cured.Plants can be protected from late tomato blight with fungicides containing maneb, mancozeb, chlorothanolil, or fixed copper."
        elif dname=='Tomato leaf mold':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Tomato leaf mold is a fungus that develops when the leaves are wet for long periods of time and the relative humidity is high (greater than 85 percent). Because of this need for moisture, the disease is mostly seen in hoophouses and greenhouses."
            control="Preventing this disease can be as simple as lowering the relative humidity in the hoophouse.Plant spacing should be increased, weeds should be removed, and trellis plants should be pruned.To reduce leaf moisture, use drip irrigation."
        elif dname=='Tomato Septoria leaf spot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Septoria leaf spot is caused by Septoria lycopersici, a fungus. It is one of the most damaging diseases to tomato foliage, and it is especially bad in locations where rainy, humid weather lasts for long periods of time. After the first fruit sets, Septoria leaf spot occurs on the lower leaves."
            control="Getting rid of diseased leaves.Use organic fungicides if possible. Fungicides containing copper or potassium bicarbonate will aid in the prevention of disease spread..Use chemical fungicides if necessary. Chlorothalonil is one of the least hazardous and most effective pesticides (sold under the names Fungonil and Daconil)."
        elif dname=='Tomato  Two spotted spider mite disease':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Two spotted spider mites are infrequent pests that can seriously harm some vegetable crops in hot, dry weather. Tomatoes, beans, muskmelons, watermelons, and sweet corn are all susceptible to mites. Mites thrive in hot, dry weather over long periods of time. Infestations commonly start near the edge of a field, near rank weed growth or dirt roads."
            control="Insecticidal soaps are quite effective and should be used as a first line of defense against spider mites in most cases.directing a strong spray of water at affected plants to remove spider mitesHorticultural oil may also be effective, but it must be used on a regular basis."
        elif dname=='Tomato Target spot':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The fungus Corynespora cassiicola causes tomato target spot. In tropical and subtropical areas of the world, the disease affects field-grown tomatoes. In 1967, the target spot was discovered on tomatoes for the first time in the United States, in Immokalee, Florida."
            control="At the end of each growing season, remove old plant detritus.To prevent the disease from spreading, keep the leaves dry.Chlorothalonil, mancozeb, and copper oxychloride-based products have been proven to effectively control target spots."
        elif dname=='Tomato mosaic virus':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Tomato mosaic virus (ToMV) can cause tomato plants to yellow and stunt, resulting in a loss of stand and lower yield. ToMV may induce irregular fruit ripening, lowering yield even more."
            control="Fungicides will not help with this viral infection.When available, plant resistant varieties or get transplants from a trustworthy supplier.Seed from contaminated crops should not be used.All contaminated plants should be removed and destroyed."
        elif dname=='Tomato yellow leaf curl virus':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Tomato yellow leaf curl virus is a DNA virus belonging to the family Geminiviridae and the genus Begomovirus. TYLCV is the most damaging tomato disease, and it can be found in tropical and subtropical areas, inflicting significant economic losses."
            control="Virus-infected plants have no treatment options.Plants should be removed and destroyed whenever possible.Controlling weeds surrounding the garden can help to prevent virus transmission by insects, as weeds are typically hosts for viruses.Control using chemicals Imidacloprid should be sprayed all over the plant, including underneath the leaves."
        elif dname=='Apple Healthy leaves':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Apple tree leaves are oval, dark green, and have an uneven leaf base. They have serrated and curled leaf edges. The branches are reddish brown in colour and The bark is gray-brown in colour and is peeling away.The flowers have five petals and are red to white in color and their blooming season is April-May  Fruit: monoecious / hermaphrodite gender distribution The color and size of the stone fruit (apple) varies widely according on the variety."
            control=""
        elif dname=='Blueberry healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="From spring to summer, blueberry plants grow beautiful leaves that are green or bluish green. In the fall, the leaves turn red or have reddish tints. The leaves are ovate, with an uneven oval or somewhat egg form that is wider at the bottom than the top. Blueberries are deciduous, meaning they lose their leaves in the late fall and early winter. Many species of bare canes reveal hues of crimson, lending decorative appeal to the winter environment."
            control=""
        elif dname=='Corn healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The leaves resemble those of the vegetable maize plant (Zea mays), hence the sometimes-used nickname 'corn leaf plant'. The leaves are bright green with strong veins running along parallel lines. As the plant ages and loses its lower leaves, the stems become more visible. Corn plant, albeit uncommon on indoor plants, can yield fragrant panicles of delicate yellow and white flowers in the winter and spring, followed by little red berries."
            control=""
        elif dname=='Grape Healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Grape leaves range in size from medium to large and are cordate, or heart-shaped, with numerous lobes. The smooth, brilliant green leaves develop in an alternating pattern with serrated, or toothed, edges and pointy points on each lobe. Grape leaves grow on climbing vines that can reach heights of over seventeen meters. To ascend, the vines employ forking tendrils, which are small branches that curl around other plants and things. Grape leaves are soft and flavorful, with a subtle lemony, green, and acidic flavor. Grape leaves are accessible from mid-summer through early winter."
            control=""
        elif dname=='Peach healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Peach leaves are tiny to medium in size and form, ranging from oval to lanceolate, and measure 10-20 centimeters in length and 2-8 centimeters in width. The smooth, bright green leaves have serrated edges that taper to a point at the non-stem end, and there is a central midrib with numerous thin veins spreading across the surface. Peach leaves are long and slender, and they grow in an alternate pattern. Peach leaves cannot be eaten raw, but when cooked, they have a slightly bitter flavor with almond and flowery notes. Peach leaves are accessible from spring to summer."
            control=""
        elif dname=='Pepper bell Healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Pepper leaves are small to medium in size and long, lanceolate, or oval in shape, averaging 5-10 cm in length. The dark green leaves grow in an alternate pattern and are smooth on top and somewhat matte and lighter green on the underside. Pepper leaves have smooth edges and grow on a shrub-like plant that yields red or green fruit. Like spinach, the leaves are best plucked when they are young and fragile. Unlike the fruit, pepper leaves have little to no heat and instead have a spicy, slightly sweet, and grassy flavor with a little bitterness and warm overtones."
            control=""
        elif dname=='Potato Healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The potato plant has a branching stem and alternately arranged leaves with unequally sized and shaped leaflets. The leaflets can range in shape from oval to oblong, and the leaves can grow to be 10–30 cm (4–12 in) long and 5–15 cm (2–6 in) wide. White or blue blooms and yellow-green berries are produced by the potato plant. Potato tubers develop underground and are typically found in the top 25 cm (10 in) of soil."
            control=""
        elif dname=='Raspberry Healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The leaves of the raspberry plant are divided into three to five leaflets that expand outward. The leaflet with the most leaflets is in the center. The leaf margins are serrated, with some being finely serrated and others being obviously jagged. Raspberry leaves are often wider than they are long, with an oblong shape rather than a spherical one. The color of the leaves ranges from light green (for immature plants) to medium dark green. The leaves curl up and turn a darker shade of brownish green in the winter."
            control=""
        elif dname=='Soyabean Healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Soybean plants are typically upright shrubs with woody stems and alternately placed leaves. The leaves have three separate leaflets that are oval or lance-like in shape and can grow to be 3–10 cm (1.2–4.0 in) long. The soybean plant produces little white or purple blooms as well as curving seed pods about 3–15 cm (1.2–6 in) in length and containing 1–5 seeds."
            control=""
        elif dname=='Strawberry Healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Strawberry leaves are small to medium in size, flat and rectangular in shape, and 4-5 cm wide on average. The matte dark green leaves feature silky hairs on the underside and notched or serrated edges. The leaves grow in threes on hairy stalks that can grow to be 10-15 cm tall. Strawberry leaves develop on a trailing plant that extends out low to the ground on horizontal runners that transform into roots, allowing new plants to sprout. The plants are also distinguished by their little white flowers and bright crimson fruits. Strawberry leaves have a light, sweet flavor with grassy, herbal undertones and a moderately astringent finish. Strawberry leaves are available in spring and summer."
            control=""
        elif dname=='Cherry healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="Sour Cherry leaves are long and oval. They're dark green and shiny, with downy undersides. Each leaf can grow to be 5 to 8 centimeters long and 3 to 7 cm wide. On the steam, there is one leaf per node, and the edge of the leaf blade may be serrated, with the leaves ending in sharp points. Fruit varies in color from vivid red to nearly black and grows on short stalks. The fruit and leaves appear on the shrubby tree, which grows to a height of 4 to 10 meters. The taste of sour cherry leaves is harsh. Sour Cherry leaves can be found throughout the spring, summer, and fall."
            control=""
        elif dname=='Tomato healthy':
            data=dname
            pic=os.path.join(app.config['upf'],'dropdown/'+ dname + '/image.JPG')
            content="The tomato plant's leaves range in length from 4 to 10 inches. Tomato plants have pinnate leaves with five to nine leaflets per petiole. Yellow flowers with five lobes are produced by the plants."
            control=""


    return render_template('document.html',li=name,d=data,con=content,cot=control,img=pic)




@app.route('/Prediction', methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        data = request.files['file'].read()
        print(type(data))
        im = base64.b64encode(data)
        img = Image.open(io.BytesIO(data))
        arr = np.array(img)

        inp = resize(arr, (224,224))
        g = inp
        print(type(inp))
        print(inp.shape)
        inp = inp.astype(np.float32)
        inp = inp.reshape((1,224,224,3))
        pred = model.predict([inp])[0]
        top3 = list(pred.argsort()[-3:][::-1])


        print(top3)


        l = list(labels.keys())
        class1 = [l[top3[0]],pred[top3[0]]*100]
        class2 = [l[top3[1]],pred[top3[1]]*100]
        class3 = [l[top3[2]],pred[top3[2]]*100]

        # convert numpy array to PIL Image


        g = g*255
        img_o = Image.fromarray(g.astype('uint8'))
        # create file-object in memory
        file_object = io.BytesIO()
        # write PNG in file-object
        img_o.save(file_object, 'PNG')
        file_object.seek(0)
        img_base64 = base64.b64encode(file_object.getvalue()).decode('utf8')

        return render_template('pred.html', base64img = img_base64,class1 = class1,class2 = class2,class3 = class3)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
