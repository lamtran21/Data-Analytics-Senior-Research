# Data-Analytics-Senior-Research
Attractiveness lies in the Averageness

## File description

camera_calibration.py, check_resources.py, facial_feature_dectector.py, frontalize.py: code to frontalize the images and get the landmarks, will be called by data_aggregation.py

data_aggregation.py: code to aggregate 3 dfs: landmarks (called the above functions), attributes (attractiveness rating, read from data source), demographic 
(age/gender/race, read from data source). This code will produce aggregated_df.csv after ~3 hours.

facical_attributes.py: reads in aggregated_df and produce facial attributes from facial landmarks (see Appendix in report). This code will produce facial_attribute.csv

random_forest.py: reads in facial_attributes.csv and run linear regression + random forest model

_49 files are files to read in 49 publication-friendly images (images that could be shown to the public as the 2,222 images in the training set cannot be) and produce Figure 1 in the report. Codes in the _49 files are similar to the main files.

## Reference

(Data) Bainbridge, W.A., Isola, P., & Oliva, A. (2013). The intrinsic memorability of face images. Journal of Experimental Psychology: General. Journal of Experimental Psychology: General, 142(4), 1323-1334. (http://www.wilmabainbridge.com/facememorability2.html)

(Frontalization) Hassner, T., Harel, S., Paz, E., & Enbar, R. (2015). Effective Face Frontalization in Unconstrained Images. Conf. on Computer Vision and Pattern Recognition. https://doi.org/10.1162/089976606774841602


