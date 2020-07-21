hwobj= jetson('192.168.51.155','usagi','0000');

envCfg = coder.gpuEnvConfig('jetson');
envCfg.DeepLibTarget = 'cudnn';
envCfg.DeepCodegen = 1;
envCfg.Quiet = 1;
envCfg.HardwareObject = hwobj;
coder.checkGpuInstall(envCfg);
%%
type resnet50_wrapper
cfg = coder.gpuConfig('exe');
cfg.Hardware = coder.hardware('NVIDIA Jetson');
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');

im=single(imread('peppers.png'));
im=imresize(im,[224,224]);

cfg.CustomSource=fullfile('main_resnet50.h');
cfg.CustomSource=fullfile('main_resnet50.cu');

codegen -config cfg -args {im} resnet50_wrapper -report
hwobj.putFile('synsetWords_resnet50.txt',hwobj.workspaceDir);
hwobj.runApplication('resnet50_wrapper');