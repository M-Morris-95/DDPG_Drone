mdl = 'Transition';
%https://harikrishnansuresh.github.io/assets/deep-rl-final.pdf
open_system(mdl)

% Specify observer information
numObs = 12;
obsInfo = rlNumericSpec([numObs 1],'LowerLimit',-1,'UpperLimit', 1);
obsInfo.Name = 'Observer';


% Specify actor information
numAct = 6;
actInfo = rlNumericSpec([numAct 1],'LowerLimit',-1,'UpperLimit', 1);
actInfo.Name = 'actor';

% Initialise environment - simulator, agent, observer information, actor
% information
env = rlSimulinkEnv('Transition','Transition/RL Agent',...
    obsInfo,actInfo);

% Set model params
Ts = 0.3; % sample time
Tf = 100;  % finish time
rng('shuffle');    % Set rng seed

env.ResetFcn = @(in)setVariable(in,'Desired_Location',40*rand(3,1)-20,'Workspace',mdl);

% Create some convolutional neural nets for the critic
% specify the number of outputs for the hidden layers.
hiddenLayerSize = 200;                               % was 80

statePath = [
    imageInputLayer([numObs 1 1], 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'CriticStateFC2')
    reluLayer('Name', 'CriticRelu2')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'CriticStateFC3')];
actionPath = [
    imageInputLayer([numAct 1 1], 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'CriticActionFC1', 'BiasLearnRateFactor', 0)];
commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'CriticCommonFC1')
    reluLayer('Name', 'CriticCommonRelu2')
    fullyConnectedLayer(1, 'Name','CriticOutput')];


criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC3','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',2,'UseDevice','gpu');
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOptions);

% Create the actor network (another CNN)
actorNetwork = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc2')   %oscillates a bit without these bois
    reluLayer('Name','relu2')                           %oscillates a bit without these bois
    fullyConnectedLayer(numAct,'Name','fc3')
    tanhLayer('Name','tanh1')];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',2,'UseDevice','gpu');
actor = rlRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'tanh1'},actorOptions);


% Create the DDPG agent
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts  ,...
    'TargetSmoothFactor',0.125,...
    'ExperienceBufferLength',1e6 ,...
    'DiscountFactor',0.98,...
    'ResetExperienceBufferBeforeTraining',0,...
    'MiniBatchSize',32);

agentOpts.NumStepsToLookAhead = 15;
agentOpts.NoiseOptions.InitialAction = 0;
agentOpts.NoiseOptions.Mean = 0;
agentOpts.NoiseOptions.Variance = 0.225;                  % tried 0.3, v slow
agentOpts.NoiseOptions.VarianceDecayRate = 0.004;
agentOpts.NoiseOptions.MeanAttractionConstant = 1;
agent = rlDDPGAgent(actor,critic,agentOpts);



maxepisodes = 10000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',15, ...                % was 20
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopOnError', 'off',...
    'StopTrainingCriteria','AverageReward',...
    'SaveAgentCriteria',"EpisodeCount",...
    'SaveAgentValue',1,...
    'StopTrainingValue',8000);


doTraining = true;

disp('set up complete')
load('Agent240.mat')
%agent = agent.saved_agent;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load pretrained agent.
end
experience = sim(env,agent);
