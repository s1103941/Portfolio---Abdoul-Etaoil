# Portfolio---Abdoul-Etaoil
Dit is mijn portfolio voor de minor Applied Data Science, Abdoul Etaoil - 15082121

# Datacamp bewijslast

Zoals in de afbeelding hieronder te zien is, heb ik alle datacamp courses behaald.
![Screenshot](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/images/datacamp.png)

# Inhoudsopgave

1. [Reflectie](#reflect)
    1. [Reflectie op eigen contributie](#contributie)
    2. [Reflectie op leerdoelen in project](#leerdoelen)
    3. [Reflectie op groepsproject als geheel](#groepsproject)
2. Research methods
    1. [Task Definition](#task)
    2. [Evaluation](#evaluation)
    3. [Conclusions](#conclusions)
    4. [Planning](#planning)
3. Predictive analytics
    1. [Selecting a model](#select)
    2. [Configuring a model](#configure)
    3. [Training a model](#train)
    4. [evaluating a model](#evaluate)
4. Data preprocessing
    1. [Data explanation](#explanation)
    2. [Data exploration & visualization](#exploration)
    3. [Data cleansing & preparation](#clean)
4. Communcatie
    1. [Presentaties](#presentaties)
    2. [Paper](#paper)
   


# Research methods

## Task Definition<a name="task"></a>

Ons doel, van het probleem van dit project was om erachter te komen of het mogelijk is, dat cybercriminaliteitsslachtoffers een aantal specifieke kenmerken of eigenschappen hebben, waardoor ze meer kans hebben om slachtoffer te worden.

Na met de groep te hebben gezeten en de dataset te hebben bekeken die we zouden gaan gebruiken, hebben we in eerste instantie besloten tot de onderzoeksvraag: “Which characteristics associate positively with (a type of) cybercrime victimization in the Netherlands?” Tijdens het project is de onderzoeksvraag meerdere malen geherformuleerd.

Nadat we zagen dat we met gegevens uit 2016 werkten, herformuleerde we de onderzoeksvraag naar "Welke economische, demografische en psychologische kenmerken associëren positief met (een vorm van) cybercriminaliteitsslachtofferschap van Nederlanders in 2016? Omdat we werkten met sociaal economische gegevens, en met een gegevensbron die op meerdere manieren te beantwoorden was, was dit de vraag die we voor ogen hadden. 

Aan het eind van het project hebben we de vraag nog een laatste keer geherformuleerd naar : "Welke economische en demografische kenmerken associëren positief met (een soort) cybercriminaliteitsslachtoffer van Nederlanders in 2016", waarbij we het psychologische deel wegnamen, omdat we erachter kwamen dat we niet zoveel informatie uit de dataset konden halen.

De deelvragen die geformuleerd zijn om antwoord te geven aan de hoofdvraag zijn : 

How should reported cybercrime victims through the scope of this project be defined?

Which socio-economic and demographic features does the dataset from SN contain?

How to account for major imbalances between datasets when it comes to training machine learning models?

Which models are appropriate for classifying cybercrime victims based on categorical values?

How can the chosen models be properly evaluated?

What method can be applied to extract features that are positively associated with those victimized by cybercrime?

How can be statistically proven that victimized groups, individuals that have a certain amount of features in common, are significantly present in the dataset?


## Evaluation<a name="evaluation"></a>

Omdat we met een kleine hoeveelheid gegevens werkten, met zeer vergelijkbare kenmerken, was het voor ons moeilijk om te zien of er karakteristieken waren die konden verklaren of iemand meer kans had om slachtoffer van cybercriminaliteit te worden. Toch konden we zien dat met 3 van onze best presterende modellen, er een aantal karakteristieken tevoorschijn kwamen die het model besloot te gebruiken, wat ook in de andere modellen te zien was. Voor toekomstig werk zou ik aanraden om een soort neurale netwerk te gebruiken, om te zien of de voorspellingen verbeteren, of om de functies die uit ons onderzoek naar voren kwamen te gebruiken, en om te proberen ze beter te evalueren.

## Conclusion<a name="conclusions"></a>

Voor de paper, heb ik de conclusie geschreven, om de hoofdvraag 'Which characteristics associate positively with (a type of) cybercrime victimization in the Netherlands?' te kunnen beantwoorden

Het doel van deze studie was om bepaalde groepen te vinden die een aantal kenmerken kunnen beschrijven die een positief verband hebben met het feit dat iemand het slachtoffer van cybercriminaliteit kan worden. Ten eerste zijn de resultaten van de classificatie-algoritmen zo'n 65% nauwkeurig gemeten, waardoor er geen specifieke claims kunnen worden gedaan ten aanzien van een concreet slachtofferschapsprofiel. Bovendien geeft de score aan dat er voor sommige groepen de neiging bestaat om slachtoffer te worden. Dit kan worden geconcludeerd omdat de modellen in staat zijn om een onderscheid te maken tussen slachtoffers en niet-slachtoffers met een nauwkeurigheid die 15% beter is dan het toevallige gissen, dit kan je zien aan de scores van de modellen in de tabel hieronder.
![Screenshot](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/images/metrics.png)


Het kenmerkende belang van de verschillende modellen kan dus nog steeds informatie geven over welke kenmerken in de zaak relevant zijn. De leeftijdsgroepen zijn belangrijk voor alle modellen en ook voor de chi-square toets. Het geslacht van het slachtoffer is ook aanwezig in sommige modellen en is een relevant splitsingspunt voor de decision tree. Verder blijkt uit de resultaten van t-SNE dat het niet in staat is om slachtoffers op hun kenmerken correct te clusteren, zoals hieronder te zien : 
![Screenshot](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/images/is_cyber_victim_1.png)

Wat betreft de sociaaleconomische en demografische kenmerken die positief samenhangen met de gerapporteerde slachtofferschap van cybercriminaliteit bij Nederlanders in 2016, beperken de wisselende resultaten van de modellen de eventuele claims die rond de kenmerken kunnen worden gedaan. Maar de score van de modellen presteert wel beter dan random, wat weer suggereert dat hier wellicht iets te vinden is. Een aanbeveling is om deep learning te gebruiken, zoals een Neural Network, en op basis hiervan kijken of deze een hogere accuracy geeft dan de modellen die wij in ons onderzoek gebruikt hebben

## Planning<a name="planning"></a>

Tijdens het project is er gebruik gemaakt van scrum, waarbij er gewerkt werd binnen sets van 2 scrums. Onderstaand is een tabel van mijn belangrijkste tickets, die ervoor gezorgd hebben om de deelvraag ‘’ te beantwoorden.

Taak | Actie
------------ | -------------
Machine Learning | Try ML classification algorithms: SVM
Machine Learning | Try ML classification algorithms: K-NN
Machine Learning | Try ML algorithm: t-SNE
Machine Learning | Try PCA

Onderstaande heb ik ook een foto toegevoegd van de scrumbord die we gebruikt hebben, waaronder 1 van bovenstaande tickets te zien is.

![Screenshot](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/images/scrumbord.png)

# Predictive Analytics
# Ik heb binnen dit hoofdstuk gewerkt met een kaggle dataset, in verband met privacy redenen van het CBS. De notebooks voor SVM en PCA/t-SNE zijn bijgevoegd aan de repository, maar zonder output. Deze is uiteraard binnen de kaggle dataset wel te zien.

## Selecting a model<a name="select"></a>

In onze tijd bij het CBS bevatte onze dataset gelabelde gegevens. Omdat onze doelvariabele een voorspeller van cybercriminaliteit was, probeerden we te zien of we deze variabele konden voorspellen op basis van de kenmerken van mensen. Aangezien dit een classificatieprobleem is (en we ons beperkten tot SKLearn), beperkt dit de keuzes van het model tot classificatie-algoritmen zoals : 

- Support Vector Machine
- Logistic Regression
- Random Forest
- K-Nearest Neighbors

Deze modellen zijn gekozen vanwege hun vermogen om het belang van functies uit te leggen. Ik heb gewerkt aan de Support Vector Machine, en K-Nearest Neighbors. SVM werd gebruikt vanwege de mogelijkheid om complexe relaties met datapunten vast te leggen. 
Bashir, A (2012) vergeleek SVM met een ander classificatie-algoritme dat vergelijkbaar is (Logistic Regression), en het SVM-algoritme presteerde beter in bepaalde situaties.

Bron : Abdallah Bashir (2012). Comparative study on classification performance between support vector machine and logistic regression. International Journal of Machine Learning and Cybernetics. 4. 10.1007/s13042-012-0068-x. 

Verder heb ik ook gewerkt met cluster algoritmen, zoals PCA, en t-SNE. Dit zijn unsupervised machine learning algoritmen, en deze werden gekozen, doordat ze populair, en veelgebruikte cluster algoritmen zijn, die wellicht in staat zouden zijn om groepen van slachtoffers bij elkaar te kunnen clusteren. Ook werden ze gekozen omdat deze 2 machine learning algoritmen ook gebruikt worden voor feature extraction, waardoor ze 2 functies dienen wat efficienter is gezien het duur van het project

## Configuring a model<a name="configure"></a>

Voor het configureren van het model heb ik gebruik gemaakt van een vorm van Cross validatie genaamd Randomized Search CV. Ik voer het algoritme, het bereik van de hyperparameters die de Machine Learning Models gebruiken, en het algoritme zal het model meerdere keren draaien met willekeurige parameters binnen het bereik, totdat het de beste set van hyperparameters vindt om te gebruiken binnen het model. In het notitieboekje heb ik dit uitgevoerd op een Support Vector Machine, door de hyperparameters : 'Gamma', 'C', en de kerneltype een random waarde te laten geven, zoals hieronder te zien is.

De K-NN algoritme dat ook in het notebook te vinden is, is ook getuned, door het algoritme meerdere keren te draaien met verschillende neighbors als parameter, en dit is gevisualiseerd in de afbeelding hieronder : 
![K-NN](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/images/k-nn.png)


## Training the model<a name="train"></a>

Na het trainen van het model met de best mogelijke hyperparameters, vergelijk ik de trein- en testgegevens met behulp van het classificatierapport van SKLearn. In eerste instantie kijk ik naar de nauwkeurigheid, om te kijken welk percentage van alle gevallen het model juist voorspelde: in dit geval is de nauwkeurigheid 78%, wat vrij goed is. Deze nauwkeurigheid geldt voor de trainingsgegevens. Voorafgaand aan de training van het model zijn de gegevens gesplitst in een trainingsset, en een testset, om te zien hoe het model presteert op ongeziene gegevens. Op de testgegevens geeft het model een nauwkeurigheid van : 74%. Het ziet er naar uit dat het model niet aan het overfitten, of underfitten is. Voor de zekerheid heb ik een ROC curve aan de notebook toegevoegd, de sample die ik heb genomen is gebalanceerd, dus de ROC curve is overbodig, maar dit zou ik kunnen gebruiken als een vorm van evalueren.

## Evaluating the model<a name="evaluate"></a>

Voor het evalueren van verschillende modellen voor de dataset, heb ik voor het project bij CBS een notebook gebouwd, wat alle machine learning modellen draait, en de scores van elke model(accuracy, precision, f1-score) opslaat. Dit model is hier te vinden. De modellen kunnen hiermee geevalueerd worden. Voor de scores verwijs ik u naar de paper, of naar de hoofdstuk research methods hier. 

De link naar de evalueer notebook is hier te vinden : [Notebook](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/notebook/AutomatedModels.ipynb)

Binnen de kaggle dataset, heb ik naast SVM, ook k-NN gedraaid, en vergelijk ik de modellen met elkaar. Na gekeken te hebben naar de scores, en door k-NN ook te hyper parameter tunen, ziet het er naar uit dat K-NN een iets betere performance. De train en test accuracy zijn ook erg dicht bij elkaar, dus het ziet er ook naar uit dat het model niet aan het overfitting of underfitten is.

## Visualizing the outcome of a model<a name="outcome"></a>

Voor SVM, heb ik een ROC curve getekend als voorbeeld van model visualisatie/evaluatie : 
![Screenshot](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/images/ROC.png)

Voor de verdere visualisatie van de modellen, verwijs ik u naar de notebook dat hier te vinden is : 


# Data preprocessing

# Data explanation<a name="explanation"></a>

## Data toelichting CBS dataset

Binnen het CBS hebben we gewerkt met 2 datasets, de sociaal economische database, wat alle personen bevat in nederland met hun sociaal-demografische gegevens/karakteristieken, en het 2016 politiebestand, welke een predictor bevat of iemand is geclassificeerd als een cybercrime slachtoffer of niet.

## Data toelichting kaggle dataset
Aangezien ik tijdens het project weinig ben betrokken bij data exploratie, zal ik een kaggle dataset gebruiken om dit te kunnen bewijzen, welke data bevat over de titanic slachtoffers, met explanatory variabelen die beschrijven of een persoon de ramp heeft overleefd, of niet. Belangrijke variabelen zijn : 

Survived : Heeft deze passagier de ramp overleefd of niet
Pclass : De klasse van de passagiers (hoe hoger de klasse, hoe luxer)
Fare	: Prijs van een kaartje
Cabin	: In welke kabine de passagier zich bevind.

[Link naar notebook](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/notebook/titanic.ipynb)

## Data exploration & Visualization<a name="exploration"></a>

## Outlier in CBS data
Nadat ik de PCA algoritme, en t-SNE algoritme heb gedraaid op de dataset, heb ik gemerkt dat er bepaalde punten zijn die ver van de rest zijn. Hiermee had ik succesvol outliers kunnen detecteren, wat vervolgens uit de dataset werd gehaald omdat het model hier een bias op kan hebben : zie het figuur hieronder voor bewijslast.

![Screenshot](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/images/pca.png)



## Data exploration kaggle dataset
Voordat ik begin met het visualiseren van de data, inspecteer ik de data om te kijken of er waardes zijn die ik niet zou verwachten(outliers), en of er waardes missen. Na een korte inspectie gedaan te hebben zie ik, dat er van 20% van de mensen de leeftijd mist, en van 77.7% van de mensen niet bekend is in welke kabine ze zaten. Ik zal deze waardes moeten vervangen, of moeten verwijderen, voordat er begonnen zal worden aan Machine Learning. Nadat de waardes zijn opgeschoond, heb ik visualizaties gemaakt, wat tot nieuwe inzichten heeft kunnen leiden met de dataset. Ik merk dat mensen die een ticket prijs van boven de 500 euro hebben, de dataset 'skewen', dit zijn outliers die ik uit het model heb verwijderd. 

## Data cleansing & Preparation<a name="clean"></a>

[Link naar notebook](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/notebook/titanic.ipynb)

Tijdens het inspecteren van de dataframe, heb ik gemerkt dat er 2 kolommen zijn die veel waardes missen. De leeftijd kolom mist ongeveer 20% van de waarden, en de kabine kolom ongeveer 80%. Ik heb ervoor gekozen om de kabine kolom weg te laten, omdat er te veel data mist. Het zou immers kunnen dat het model de kolom gebruikt om te bepalen of iemand de ramp heeft overleefd, wat niet de bedoeling is omdat 80% van de mensen deze waarden missen. Voor de leeftijd kolom is er gekozen om de missende waardes te imputeren, hier is voor gekozen omdat er niet teveel data mist, en omdat leeftijd een belangrijke factor kunnen zijn om te overleven(de kinderen eerst). Tijdens data exploration heb ik verder gemerkt dat er outliers in de data zaten, dit waren mensen die een ticket prijs hadden die te hoog was. Nadat de data is schoongemaakt, heb ik de dataframe nogmaals geinspecteerd. Binnen de dataset, zitten er een paar variabelen die categoriaal zijn. 

Aangezien Machine Learning niet goed overeen kan met tekst, heb ik ervoor gekozen om de data te transformeren door middel van ‘One Hot Encoding’. Dit geeft elke categorie een binaire waarde in een aparte kolom. Vervolgens inspecteer ik de dataframe nogmaals, en is er nog 1 stap dat moet gebeuren. Aangezien er kolommen zijn met uiteenlopende waarden, zoals de ticket prijs, en leeftijd, zou het mogelijk zijn dat een model een bias zal krijgen tegenover deze waarden, wat tot minder accurate predicties lijkt. Ik kies er voor om de data opnieuw te transformeren, dit keer om het te standardiseren op een normale verdeling, zodat alle waarden even gedistribueerd zijn. De data is naar mijn mening nu geschikt voor Machine Learning.


# Communicatie

## Presentaties<a name="Presentaties"></a>

Tijdens de minor heb ik 4 presentaties gegeven, waarvan 2 presentaties intern zijn, en 2 extern. Mijn eerste presentatie is gegeven tijdens de tweede week van September, tijdens deze presentatie heb ik een introductie gegeven over ons project, en onze progressie tot dan.

De tweede presentatie die ik heb gegeven, was een externe presentatie op 28 oktober. Tijdens deze presentatie heb ik het gehad over onze exploratory data analysis, feature elimination en feature extraction, en heb ik wat meer duidelijkheid gegeven over onze definitie van een cybercrime slachtoffer. Verder heb ik ook de PCA en t-SNE plots gepresenteerd, die ik heb ontwikkeld tijdens het project. Zie : [Github](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/presentations)

De derde presentatie was een externe presentatie, op 29 november die ik samen heb gegeven met Nicky. Hierbij heb ik de recap gepresenteerd van de laatste externe presentatie, om wat duidelijkheid te geven aan het publiek.

De laatste presentatie die ik heb gegeven, was een interne presentatie in December. Tijdens deze presentatie heb ik het gehad over de machine learning modellen die we tot nu toe behandeld hadden, en welke het er best uit is gekomen.

Zie deze folder voor de presentaties die ik heb gegeven : [Github](https://github.com/s1103941/Portfolio---Abdoul-Etaoil/blob/master/presentation)

# Schrijven van de paper<a name="paper"></a>

In de paper heb ik binnen de ‘Models’ hoofdstuk, beschreven hoe Support Vector Machines, en t-SNE werken(Ik heb deze models gemaakt), en waarom we gekozen hebben voor deze models. Ook heb ik beschreven waar we deze models voor gebruiken. Verder heb ik de conclusie geschreven binnen de paper, naar aanleiding van de notebook die ik heb ontwikkeld dat alle algoritmen evalueert.



# Reflectie <a name="reflect"></a>

# Reflectie op eigen contributie in project<a name="contributie"></a>

## Situatie

Ik doe mee aan de minor Applied Data Science aan de Haagse Hogeschool, hiervoor ben ik geplaats in een groep met 4 andere mensen, en zal ik onderzoek doen om een vraag te beantwoorden voor het Centraal Bureau voor de Statistiek. Om deze vraag te kunnen beantwoorden zal ik Data Science, en om wat specifieker te zijn Machine Learning moeten toepassen.

## Taak

Aangezien ik een achtergrond heb in Software Engineering, zag ik het als mijn rol om te programmeren binnen de groep. Ik heb al een achtergrond in Python, waardoor ik geen vertraging had met het leren van programmeren.  Door het project heen heb ik veel taken gehad, in het begin van het project heb ik me gefocust op exploratory data analysis, ik heb de datasets waar we mee werkte geinspecteerd, en heb op basis hiervan eerste grafieken gemaakt wat tot meer inzichten heeft kunnen leiden.

Vervolgens heb ik als taak opgenomen, om unsupervised machine learning algoritmen te kunnen gebruiken, om te kijken of deze algoritmen groepen van personen kunnen clusteren op basis van de karakteristieken, om te kunnen kijken of deze groepen een grotere kans hebben om slachtoffer te worden van een cybercrime.

Na het bouwen en implementeren van t-SNE, en PCA heb ik me gefocust op supervised Machine Learning algoritmen, en heb 2 algoritmen(KNN,SVM) gedraaid op de dataset, om te kijken of er karakteristieken zijn die de algoritmen gebruiken om cybercrime slachtoffers te voorspellen.

Als laatste heb ik me gefocust op het evalueren van alle modellen, en heb ik een notebook ontwikkeld waarin alle modellen staan die we ontwikkeld hebben binnen de groep. Vervolgens heb ik op basis van deze informatie een conclusie getrokken, en dit opgenomen in de paper.

## Actie

Tijdens het inspecteren van de data, en het doen van Explanatory Data Analysis, heb ik een cursus gevolgd op Datacamp. De eerste fase was dus vooral het programmeren, en toepassen van de opgedane kennis uit de Datacamp course op de dataset. 

Vervolgens heb ik literatuuronderzoek gedaan naar clusteralgoritmen, en ben ik hier gekomen op t-SNE, en PCA. Ik heb de datacamp course gevolgd voor deze algoritmen, en ben op zoek geweest naar artikelen dat deze goed beschrijven. Vervolgens heb ik deze toegepast, en heb ik dit in de tweede externe presentatie behandeld. 

Daarna heb ik onderzoek gedaan naar de supervised machine learning algoritmen : KNN en SVM, en heb ik deze toegepast op de dataset. Als laatste heb ik mijn model geevalueerd, een notebook gebouwd dat alle modellen opneemt, die tegenelkaar geevalueerd werden, en gewerkt aan de paper

## Resultaat

Tijdens Exploratory Data analysis, heb ik plots kunnen leveren dat diende als inspiratie voor medegroepsgenoten. Het CBS had veel statistieke vragen, en dit is dan ook als taak opgenomen door Rik, uit ons groepje.

De unsupervised Machine Learning algoritmen produceerden grafieken, waarbij er clusters te zien waren, dat aan het project kon contributeren. Vervolgens heb ik K-NN en SVM plots gemaakt, waarbij elk model de belangrijkste features kon laten zien. Als laatste heb ik de conclusie hoofdstuk geschreven in de paper, op basis van de modellen die alle groepsgenoten hebben ontwikkeld en de evaluatie.

## Reflectie

Uiteindelijk ben ik met het grootste gedeelte van mijn resultaten niet tevreden, ik heb gemerkt dat ik het EDA snel had gedropt, terwijl ik hier wellicht meer een bijdrage aan kon leveren.

Verder heb ik vol trots de t-SNE, en PCA plots gepresenteerd, maar ben er later achtergekomen dat er bij deze plots veel informatie verloren ging, waardoor de betekenis van de clusters niet reliable waren. Ook had ik gemerkt aan het begin van het project, dat ik veel meer op de theorie van Machine Learning ging focussen, wat er voor zorgde dat ik vertraging had bij het leveren van de resultaten.

Voor de supervised machine learning algoritmen, ben ik blij met het resultaat, het kwam enigzins overeen met de modellen van mijn groepsgenoten, wat betekent dat het goed werkte. In de toekomst zal ik erop letten om de focus meer te leggen op resultaten.

# Reflectie op leerdoelen in project<a name="leerdoelen"></a>

## Situatie

Ik doe mee aan de minor Applied Data Science aan de Haagse Hogeschool, hiervoor ben ik geplaats in een groep met 4 andere mensen, en zal ik onderzoek doen om een vraag te beantwoorden voor het Centraal Bureau voor de Statistiek. Tijdens de eerste week van de minor, heb ik een paar leerdoelen voor mezelf opgesteld, die ik van plan was om te ontwikkelen/verbeteren voor het einde van de minor.

## Taak

De eerste leerdoel die ik voor mezelf heb opgesteld, is beter worden in presenteren. Tijdens de minor worden er elke week presentaties gehouden, en dat leek me de beste mogelijkheid om het te kunnen verbeteren.

De tweede, en tevens de belangrijkste leerdoel die ik voor mezelf heb opgesteld : het kunnen toepassen van Machine Learning, op echte voorbeelden in het bedrijfsleven. Dit is erg belangrijk voor mij, omdat ik van plan ben om na mijn hbo opleiding een master opleiding te volgen in computer science/data science, en zag deze minor als een voorproef om de beslissing te kunnen maken, of het een geschikte keuze is.

De derde, en laatste leerdoel die ik voor mezelf heb opgesteld is het doen van onderzoek. Tijdens mijn hbo-opleiding heb ik niet veel onderzoek gedaan, en heb ik nooit eerder een paper geschreven. Sinds onderzoek doen een belangrijk onderdeel is van een universitaire opleiding leek mij dit de beste kans.

## Actie

Om ervoor te zorgen dat mijn presentatie vaardigheden verbeterd worden, heb ik ervoor gekozen om aan 2 externe presentaties deel te nemen, en aan 2 interne presentaties. Voor het toepassen van Machine Learning, heb ik 4 verschillende machine learning modellen geleerd, en toegepast op onze data. Verder heb ik veel literatuuronderzoek moeten doen voor de paper, naar de algoritmen die ik gebruikt heb, en waarom deze algoritmen geschikt zijn. Mijn gedeelte van de paper schrijven heeft naar mijn mening erg bijgedragen aan het doen van onderzoek.

## Resultaat

Ik heb 4 presentaties ontwikkeld en gegeven aan de klas in de eerste week, 28 oktober, 29 november, en de laatste week van December, wat bijgedragen heeft aan mijn presentatievaardigheden. Voor het leren van Machine Learning heb ik 4 algoritmen ontwikkeld, en uitgevoerd, waarvan er uiteindeijk 2 zijn gekozen voor de paper. Voor mijn onderzoeksvaardigheden heb ik hier een flinke bijdrage aan kunnen leveren door mijn modellen op te zoeken naar papers, hier is uiteindelijk mijn gedeelte in de modellen sectie van de paper gekomen, en de conclusie. 

## Reflectie

Ik ben erg tevreden over het resultaat op mijn leerdoelen, de presentaties gingen naar mijn mening erg vloeiend, en ik kon de meeste vragen van het publiek beantwoorden. Dit gaf mij erg veel zelfvertrouwen, en ik zal deze vaardigheid waarschijnlijk vaak moeten toepassen in het bedrijfsleven. 

Voor Machine Learning ben ik ook erg tevreden, voordat ik begon aan de minor was ik nog niet zeker om een master te doen in Data Science, maar na het leren en toepassen van de modellen op de dataset, en het zien van de resultaten, vindt ik dit toch een leuk en goed vakgebied waar ik me verder in zou willen specialiseren.

Onderzoek had naar mijn gevoel beter gekund, ik had niet al teveel in de paper geschreven, echter vind ik toch dat ik mijn leerdoel heb bereikt, omdat ik nooit eerder literatuur onderzoek heb gedaan voor een paper, en ik dit toch tijdens deze minor geleerd heb.

# Reflectie op het groepsproject als geheel<a name="groepsproject"></a>

## Situatie

Binnen de minor Applied Data Science aan de Haagse Hogeschool, ben ik in een groep ingedeeld met 4 andere personen. Samen hebben wij een vraagstuk van het Centraal Bureau voor de Statistiek beantwoord, en hebben we over het verloop van 5 maanden een project uitgevoerd, en zal ik het verloop van dit project evalueren.

## Taak

Aan het begin van het project, hebben we als groep een research proposal ontwikkeld, en hebben we de hoofd en deelvragen bedacht, die we in onze paper zullen beantwoorden. 

Vervolgens zijn we aan het werk gegaan met het schoonmaken van de data, en heeft een gedeelte van de groep domein onderzoek gedaan naar cyber criminaliteit, om wat comfortabeler te worden met de definities.

Na het schoonmaken van de data hebben we als groep, supervised en unsupervised machine learning algoritmen toegepast op de dataset, en hebben we plots/statistieken geleverd aan het CBS dat bepaalde vragen beantwoord, dat ze voor ons hadden.

Als laatste zijn we als groep bezig geweest met het schrijven van de paper, en hebben we geprobeerd om onze machine learning algoritmen te optimaliseren

## Actie

Binnen de eerste fase van het project, toen we nog geen toegang hadden tot de CBS data, hebben we met zijn allen gewerkt aan het research proposal. Nadat dit was gefinaliseerd, en we op locatie konden werken bij het CBS hebben we het werk opgesplitst door onze groepsgenoten op basis van hun kwaliteiten. Ik heb samen met Rik de eerste notebooks geprogrammeerd, wat visualisaties gemaakt, en de rest van het groepje heeft zich gefocusd op domein onderzoek, en machine learning orientatie.

Vervolgens zijn we bezig geweest met machine learning, waarbij ik unsupervised machine learning deed, en david supervised machine learning. Rik heeft zich toen gefocust op Explanotary Data Analysis. De rest van het groepje heeft een eerste opzet van de paper bedacht, en heeft literatuuronderzoek uitgevoerd, om het probleem beter te begrijpen

Het laatste onderdeel waar we met zijn allen aan hebben gezeten, is het schrijven van de paper. Iedereen heeft een eigen stuk toegewezen gekregen, en we hebben de verschillende stukken van elkaar gereviewed.

## Resultaat

Uit het project hebben we met success een research proposal opgezet, voor het CBS. Verder heeft Rik een flinke notebook kunnen opleveren met verschillende grafieken en inzichten, voor ons maar ook voor het CBS. Unsupervised Machine Learning algoritmen bleken niet goed te werken, de resultaten daarvan vielen tegen(vanwege low variance). Voor supervised machine learning algoritmen was het resultaat niet geweldig, maar goed genoeg om een conclusie van te trekken. Uit het groepsproject is verder een paper uitgekomen, waar alle resultaten van het onderzoek aanbod komen, en een conclusie die hierop is getrokken. 

## Reflectie

Naar mijn mening ging het begin van het project soepel, maar naarmate we verder gingen in het project ging het minder goed. We hebben bijvoorbeeld : een lang gedeelte van de tijd gestoken in literatuur onderzoek doen, en naar unsupervised machine learning algoritmen, en zijn pas begonnen met Machine Learning in November. Ik denk data als we meer tijd hadden voor Machine Learning, we waarschijnlijk betere resultaten hadden kunnen krijgen, er zijn voorbeeld beeld : algoritmen die we wouden gebruiken, maar niet mogelijk was door dat het te lang duurt op de CBS server, en we de tijd hiervoor niet hadden.

Voor de paper ben erg tevreden, de paper heeft een goede structuur, en op basis van de data die we hebben verkregen en de tijd die we beschikbaar hadden, hadden we een conclusie kunnen trekken, hoewel die niet ideaal is. De samenwerking tussen alle groepsgenoten ging naar mijn mening erg goed, de taken werden verdeeld op basis van iedereen zijn skills, en we hebben toch met elkaar binnen een korte tijd resultaten kunnen boeken.

