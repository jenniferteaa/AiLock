# AiLock
This project implements AiLock, an alternative to biometrics proposed and published in the research paper 'A Secure Mobile Authentication Alternative to Biometrics.

ai.lock is introduced as a secret image based authentication method for mobile devices which uses an imaging sensor to reliably extract authentication credentials similar to biometrics. In this project, we Implement ai.lock and test its efficiency particularly its Error Tolerance Threshold (ETT).

# Error Tolerance Threshold
ETT in terms of ai.lock represents the threshold used to separate valid from invalid authentication samples. Lower τ values indicate a stricter threshold for accepting matches between images. Given a reference image (the user initially sets as the password) and a candidate image (the image that the user has clicked in order to pass the authentication) Error Tolerance Threshold serves as a critical parameter in determining the level of similarity required between reference and candidate images for them to be considered a match. Adjusting this threshold allows system designers to balance between accommodating variations and maintaining precision in image recognition tasks.
