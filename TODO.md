# TODO

- [ ] add training for commands like "stop" and "look to left/right/above/below face", "look up"
- [ ] consider RL finetuning after behavioral cloning training
- [ ] clean up unused dirs and code
- [ ] implement DAgger to improve error recovery & overall model quality
- [ ] sometimes the perturbations of faces make them go slightly out of view of the camera. this can cause issues since the vla should only be trained to track faces in the viewport. clamp perturb movements to ensure they're within cam 
- [ ] switch to 120* cam from 60* - update boundaries / sim accordingly
