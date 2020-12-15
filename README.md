# attack_honeypot_defense
Attacking and examining methods from 'Gotta Catch 'Em All: Using Honeypots to Catch Adversarial Attacks on Neural Networks.' (Shan et al 2020.  Only single label defenses are considered.  Future research will look at applying these methods and attacks to the multi-label defense presented in the paper.

'Trigger_Placement_and_Transformation_Robustness' examines the effects of backdoor trigger placement on the honeypot defense.  Additionally, rotationally robust adversarial examples are attempted as a way to bias the attack away from the trapdoors.

'Gradient Probe' presents two new algorithms that attempt to find adversarial examples without getting caught by the honeypot defense.  The first uses an initial probe step of projected gradient descent to encourage the algorithm to avoid trapdoors, and the second reduces the highest magnitude components of the gradient in projected gradient descent.
