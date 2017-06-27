let images = [
    'Elena',
    'Arjen',
    'Monalisa',
    'Trump',
    'Circle',
    'Pontdugard',
    'Alps',
    'Clown',
    'Scream'
];

let MIN_PATCH_SIZE = 5;
let MAX_PATCH_SIZE = 25;
let IMAGE_SIZE = 200
let NUMBER_OF_PARENTS = 2;
let ELITE = 5;
let TOURNAMENT_SIZE = 10;

function createData() {
    window.chart = new google.visualization.LineChart(document.getElementById('chart'));
}

google.charts.load('current', {'packages':['corechart']});
google.charts.setOnLoadCallback(createData);


let sourceImageSelector = document.getElementById('source-image-selection');
let targetImageSelector = document.getElementById('target-image-selection');

for (image of images) {
    let soption = document.createElement('option');
    soption.value = image.toLowerCase() + '.png';
    soption.text = image;
    let toption = document.createElement('option');
    toption.value = image.toLowerCase() + '.png';
    toption.text = image;
    sourceImageSelector.appendChild(soption);
    targetImageSelector.appendChild(toption);
}

function loadImage(e) {
    let image = new Image();
    let el = document.getElementById(e + '-image-selection');
    let imageName = el.options[el.selectedIndex].value;
    image.onload = () => {
        image.width = IMAGE_SIZE;
        image.height = IMAGE_SIZE;
        let canvas = e == 'source' ? srcCanvas : destCanvas;
        let ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, image.width, image.height);
        let img = new Img(ctx, IMAGE_SIZE, IMAGE_SIZE, ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE));
        if (e == 'source') {
            sourceImage = img;
        } else {
            destImage = img;
        }
    }
    image.src = imageName;

}

function randomRange(min,max) {
    return Math.floor(Math.random() * (max-min) + min);
}

class SubRegionCrossover{
	
	constructor(parent1,parent2){
		this.child1 = parent1.copy();
		this.child2 = parent2.copy();
		this.width = parent1.image.width;
		this.height = parent1.image.height;
	}
	
	crossover() {
		let patchWidth = this.width/2;
		let patchHeight = this.height;
		let xSource = 0;
        let ySource = 0;
        let xDest = 0;
		let yDest = 0;
		for(let yy = 0; yy < patchHeight; yy++) {
			for (let xx = 0; xx < patchWidth; xx++) {
                let temp1 = this.child2.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 0];
                let temp2 = this.child2.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 1];
                let temp3 = this.child2.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 2];
				this.child2.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 0] = this.child1.getImage().data[((xx+xSource) + (yy+ySource) * IMAGE_SIZE) * 4 + 0]
                this.child2.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 1] = this.child1.getImage().data[((xx+xSource) + (yy+ySource) * IMAGE_SIZE) * 4 + 1]
                this.child2.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 2] = this.child1.getImage().data[((xx+xSource) + (yy+ySource) * IMAGE_SIZE) * 4 + 2]
				
                this.child1.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 0] = temp1;
				this.child1.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 1] = temp2;
                this.child1.getImage().data[((xx+xDest) + (yy+yDest) * IMAGE_SIZE) * 4 + 2] = temp3;
			}
		}
		return [this.child1,this.child2]
	}
}

class SubRegionMutation {
    constructor(minSize, maxSize, sourceImage) {
        this.minSize = minSize;
        this.maxSize = maxSize;
        this.width = sourceImage.width;
        this.height = sourceImage.height;
        this.sourceData = sourceImage.data;
    }

    mutate(image) {
        let patchWidth = randomRange(this.minSize, this.maxSize);
        let patchHeight = randomRange(this.minSize, this.maxSize);
        let xSource = randomRange(0, this.width - patchWidth);
        let ySource = randomRange(0, this.height - patchHeight);
        let xDest = randomRange(0, image.width - patchWidth + 1);
        let yDest = randomRange(0, image.height - patchHeight + 1);

        for (let yy = 0; yy < patchHeight; yy++) {
            for (let xx = 0; xx < patchWidth; xx++) {
                image.data[((xx+xDest) + (yy+yDest) * image.width) * 4 + 1] = this.sourceData[((xx+xSource) + (yy+ySource) * image.width) * 4 + 1]
                image.data[((xx+xDest) + (yy+yDest) * image.width) * 4 + 2] = this.sourceData[((xx+xSource) + (yy+ySource) * image.width) * 4 + 2]
                image.data[((xx+xDest) + (yy+yDest) * image.width) * 4 + 0] = this.sourceData[((xx+xSource) + (yy+ySource) * image.width) * 4 + 0]
            }
        }

    }
}

class Img {

    constructor(ctx, width, height, imageData) {
        this.imageData = imageData;
        this.data = imageData.data;
        this.width = width;
        this.height = height;
        this.ctx = ctx;
    }

    show() {
        this.ctx.putImageData(this.imageData, 0, 0);
    }

    copy() {
        let imageData = this.ctx.createImageData(this.width, this.height)
        for (let i = 0; i < this.imageData.data.length; i+=4) {
            imageData.data[i] = this.imageData.data[i];
            imageData.data[i+1] = this.imageData.data[i+1];
            imageData.data[i+2] = this.imageData.data[i+2];
            imageData.data[i+3] = 255
        }
        let newImage = new Img(this.ctx, this.width, this.height, imageData);
        return newImage;
    }

}

class GeneticAlgorithm {

    constructor(populationSize, fitnessFunction, targetImage, sourceImage, attemptImage, mutationRate, crossoverRate, selectionAlgo) {
		this.populationSize = populationSize;
		this.fitnessFunction = fitnessFunction;
		this.targetImage = targetImage;
		this.population = new Population(attemptImage);
        this.sourceImage = sourceImage;
		this.mutationRate = mutationRate;
		this.crossoverRate = crossoverRate;
        this.numberOfParents = NUMBER_OF_PARENTS;
        this.numberOfChildren = this.populationSize/this.numberOfParents;
        this.iteration = 0;
        if (selectionAlgo == 'tournament')
            this.doSelection = this.tournamentSelection;
        else if (selectionAlgo == 'roulette')
            this.doSelection = this.rouletteSelection;
        else if (selectionAlgo == 'ranking')
            this.doSelection = this.rankingSelection;
		this.initializePop();
		this.evaluatePop();
    }

    run() {
		this.evaluatePop();
		let bestSolution = this.getBestSolution();
		bestSolution.img.show()
    }

    doIteration() {
            this.iteration++;
			let sortedIndividuals = this.population.getSortedIndividuals();
			let children = [];
			for (let i=0; i< ELITE; i++){
				let child = sortedIndividuals[i];
				child.setId(children.length+1);
				children.push([children.length+1,sortedIndividuals[i]]);
			}
			let parents = [];
			for(let i = 0; i<(this.populationSize/2); i++) {
				let couple = this.doSelection();
				parents.push(couple);
			}
			
			
			for(let i = 0; i< parents.length; i++) {
				let [parent1, parent2] = parents[i];
				let prob = Math.random();
				let child1 = null;
				let child2 = null;
				if (prob <= this.crossoverRate) {
					[child1, child2] = this.crossover(parent1,parent2);
				}
				else {
					[child1, child2] = [parent1,parent2];

				}
				prob = Math.random();
				if (prob <= this.mutationRate) {
					let copyAttemptImage = child1.getImage().copy();
					this.mutate(copyAttemptImage);
					let child = new Individual(children.length+1,copyAttemptImage,child1.getFitness());
					if (children.length < this.populationSize){
						children.push([children.length+1,child]);
					}
				}
				else {
					if (children.length < this.populationSize){
                        let copyAttemptImage = child1.getImage().copy();
                        let child = new Individual(children.length+1,copyAttemptImage,child1.getFitness());
						children.push([children.length+1, child]);
					}
				}
				prob = Math.random();
				if (prob <= this.mutationRate) {
					let copyAttemptImage = child2.getImage().copy();
					this.mutate(copyAttemptImage);
					let child = new Individual(children.length+1,copyAttemptImage,child2.getFitness());
					if (children.length < this.populationSize) {
						children.push([children.length+1,child]);
					}
				}
				else {
					if (children.length < this.populationSize){
                        let copyAttemptImage = child2.getImage().copy();
                        let child = new Individual(children.length+1,copyAttemptImage,child2.getFitness());
						children.push([children.length+1, child]);
					}
				}
			}

			this.population.setPopulation(children);
			this.evaluatePop();
			let bestSolution = this.getBestSolution();
			bestSolution.img.show();
    }
	
	crossover(parent1,parent2) {
		let crossoverComputer = new SubRegionCrossover(parent1,parent2);
		let [child1,child2] = crossoverComputer.crossover();
		return [child1,child2];
	}

	mutate(copyAttemptImage) {
		let subRegionMutator = new SubRegionMutation(MIN_PATCH_SIZE, MAX_PATCH_SIZE, this.sourceImage);
		subRegionMutator.mutate(copyAttemptImage)
	}
	
    rankingSelection() {
        let rankList = [];
        let individuals = this.population.getSortedIndividuals();
        let parents = [];

        for (let i = 0; i < individuals.length; i++) {
            rankList.push(i);
        }
        
        for (let i = 0; i < this.numberOfParents; i++) {
            let idx = this.selectByProportion(rankList);
            parents.push(individuals[idx]);
            individuals.splice(idx, 1);
            rankList.splice(idx, 1);
        }

        return parents;
    }

    rouletteSelection() {
        let totalFitness = 0;
        let proportionList = [];
        let individuals = this.population.getSortedIndividuals();
        let parents = [];

        for (let i = 0; i < individuals.length; i++) {
            totalFitness += individuals[i].getFitness();   
        }
        for (let i = 0; i < individuals.length; i++) {
            proportionList.push(individuals[i].getFitness() / totalFitness);
        }
        
        for (let i = 0; i < this.numberOfParents; i++) {
            let idx = this.selectByProportion(proportionList);
            parents.push(individuals[idx]);
            individuals.splice(idx, 1);
            proportionList.splice(idx, 1);
        }

        return parents;

    }

    selectByProportion(proportionList) {
        let sum = 0;
        for (let i = 0; i < proportionList.length; i++) {
            sum += proportionList[i];
        }
        let randomNumber = sum * Math.random();
        let choice = -1;
        for (let i = 0; i < proportionList.length; i++) {
            randomNumber -= proportionList[i];
            if (randomNumber <= 0) {
                choice = i ;
                break; 
            }
        }
        return choice;
    }

    tournamentSelection() {
        let individuals = this.population.getSortedIndividuals();
        let parents = [];

		for (var i = 0; i < this.numberOfParents; i++) {
            let gladiators = [];
            let gladiatorIDXs = [];
		    for (var j = 0; j < TOURNAMENT_SIZE; j++) {
                let randomNumber = randomRange(0, this.populationSize);
                gladiators.push(individuals[randomNumber].getFitness());
                gladiatorIDXs.push(randomNumber);
            }
            let max = gladiators[0];
            let maxIDX = gladiatorIDXs[0]
            for (var k = 0; k < gladiators.length; k++) {
                if (gladiators[k] > max) {
                    max = gladiators[k];
                    maxIDX = gladiatorIDXs[k];
                }
            }
            parents.push(individuals[maxIDX]);
        }
        return parents
    }

	initializePop() {
		this.population.initializePop(this.populationSize);
		let individuals = this.population.getPopulation();
		for(var i = 0; i < individuals.length; i++) {
			let copyAttemptImage = individuals[i][1].getImage();
			this.mutate(copyAttemptImage);

		}
	}
	
	evaluatePop() {
		let individuals = this.population.getPopulation();
		for (var i = 0; i < individuals.length; i++) {
			let individual = individuals[i][1];
			let individualImage = individual.getImage();
			let individualFitness = this.fitnessFunction(this.targetImage,individualImage);
			individual.setFitness(individualFitness);
			if (individualFitness > this.population.max.fitness) {
				this.population.setPopulationMax(individualFitness,individualImage);
			}
		}
	}
	
	getBestSolution() {
		// bepaal beste resultaat in de populatie
		return this.population.getPopulationMax();
	}
	
}

class Population {
	
	constructor(attemptImage) {
		this.individuals = [];
		this.attempt = attemptImage;
		this.max = {img: this.attempt, fitness: 0};
	}
	
	initializePop(populationSize) {
		for (var i = 1; i <= populationSize; i++) {
			var individual = new Individual(i,this.attempt.copy(),0);
			let entry = [i,individual]
			this.individuals.push(entry);
		}
	}
	
	getPopulation() {
		return this.individuals;
	}
	
	setPopulation(individuals) {
		this.individuals = individuals;
	}
	
	setPopulationMax(maxFitness,img) {
		this.max.fitness = maxFitness;
		this.max.img = img;
	}
	
	getPopulationMax() {
		return this.max;
	}
	
	sortFitnessScores(){
		let scores = [];
		for (let i = 0; i < this.individuals.length; i++) {
			let score = this.individuals[i][1].getFitness();
			let id = this.individuals[i][1].getId();
			let entry = [id,score];
			scores.push(entry)
		}
		scores = scores.sort(function(a,b){return a[1]>b[1];});
		return scores;
	}
	
	getSortedIndividuals() {
		let sortedIndividuals = [];
		let sortedScores = this.sortFitnessScores();
		for(let i = 0; i < sortedScores.length; i++) {
			let [id, score] = sortedScores[i];
			for(var j = 0; j < this.individuals.length; j++) {
				let [individual_id,individual] = this.individuals[j];
				if (id == individual_id) {
					sortedIndividuals.push(individual);
				}
			}
		}
		return sortedIndividuals;
	}
	
}

class Individual{
	
	constructor(id, copyAttemptImage, fitness) {
		this.id = id;
		this.image = copyAttemptImage;
		this.fitness = fitness;
	}
	
	
	getId() {
		return this.id;
	}
	
	setId(id) {
		this.id = id;
	}
	
	setFitness(newFitness) {
		this.fitness = newFitness;
	}

	getFitness() {
		return this.fitness;
	}
	
	getImage() {
		return this.image;
	}
	
	setImage(img) {
		this.image = img;
	}
	
    copy() {
        return new Individual(this.id, this.image.copy(), this.fitness);
    }
}

function distance(img1, img2) {
    let distance = 0;
    for (let i = 0, len=img1.data.length; i < len; i+=4) {
        distance += Math.abs(img1.data[i] - img2.data[i]);
        distance += Math.abs(img1.data[i+1] - img2.data[i+1]);
        distance += Math.abs(img1.data[i+1] - img2.data[i+2]);
    }
    return distance;
}

function fitness(img1, img2) {
    let fitness = 1 - distance(img1,img2)/(255*3*img1.height*img1.width);
    return fitness;
}

let srcCanvas = document.getElementById('sourceImage');
let attemptCanvas = document.getElementById('attempt')
let destCanvas = document.getElementById('destImage');

let width = IMAGE_SIZE;
let height = IMAGE_SIZE;

// meuk code om de plaatjes in te laden, roept main aan als het klaar is
let arjen = new Image();
let elena = new Image();
arjen.onload = () => {
    let ctx = srcCanvas.getContext('2d');
    srcCanvas.width = width;
    srcCanvas.height = height;
    arjen.width = width;
    arjen.height = height;
    ctx.drawImage(arjen, 0, 0, width, height);

    attemptCanvas.width = width;
    attemptCanvas.height = height;

    window.sourceImage = new Img(ctx, width, height, ctx.getImageData(0, 0, width, height));

    elena.onload = () => {
        let ctx = destCanvas.getContext('2d');
        destCanvas.width = width;
        destCanvas.height = height;
        elena.width = width;
        elena.height = height;
        ctx.drawImage(elena, 0, 0, width, height);
        attemptCtx = attemptCanvas.getContext('2d');
        attemptCtx.fillRect(0, 0, width, height);
        attemptImage = new Img(attemptCtx, width, height, attemptCtx.getImageData(0, 0, width, height));

        window.destImage = new Img(ctx, width, height, ctx.getImageData(0, 0, width, height));
        main();
    };
    elena.src = 'monalisa.png';

};
arjen.src = 'scream.png';



function main() {
    // mutation = new SubRegionCrossover(new Individual(0, sourceImage, 1), new Individual(3, destImage, 2));
    // [window.child1, window.child2] = mutation.crossover();
}

function showPlot() {
    let data = new google.visualization.DataTable();
    data.addColumn('number', 'Iteration');
    data.addColumn('number', 'Fitness');
    let exportList = [];
    let i = 0;
    while (fitnesses[i] != 0) {
        data.addRow([i, fitnesses[i]]);
        exportList.push(fitnesses[i]);
        i++;
    }

    document.getElementById('export-text').value = '['+exportList.toString()+']';

    let chartOptions = {
        width: 500,
        height:400,
        hAxis: {
            title: 'Iteration'
        },
        vAxis: {
            title: 'Fitness'
        }
    }

    chart.draw(data, chartOptions)
}

fitnesses = new Float32Array(100000);

function startAlgorithm() {
    let iterationNumber = document.getElementById('iteration-number');
    let allTimeBest = document.getElementById('all-time-best');
    let algoElement = document.getElementById('selection-algorithm');
    let selectionAlgo = algoElement.options[algoElement.selectedIndex].value;
    let populationSize = document.getElementById('population-size').value;
    let mutationRate = document.getElementById('mutation-rate').value;
	let crossoverRate = 0.8;
	ga = new GeneticAlgorithm(populationSize,fitness,destImage,sourceImage,attemptImage,mutationRate, crossoverRate, selectionAlgo);
    let runAlgorithm = () => {
        ga.doIteration();
        iterationNumber.innerHTML = ga.iteration;
        let fitness = ga.getBestSolution().fitness;
        allTimeBest.innerHTML = fitness;
        fitnesses[ga.iteration - 1] = fitness;
        if (ga.iteration < 15000) {
            window.requestAnimationFrame(runAlgorithm);
        }
    }
    runAlgorithm();
}
