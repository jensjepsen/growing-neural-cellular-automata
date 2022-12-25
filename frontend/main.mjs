

let click = null;
let session = null;

let divIters = document.getElementById('iters')

function drawCanvas (rgb, index) {
    // create an offscreen canvas
    var canvas=document.getElementById("canvas");
    var ctx=canvas.getContext("2d");

    // size the canvas to your desired image
    const images = rgb.dims[0]
    const width = rgb.dims[2]
    const height = rgb.dims[3]
    const channels = rgb.dims[4]
    const targetChannels = 4
    const pixelCount = width * height * channels
    const offset = index * width * height * channels
    canvas.width = width;
    canvas.height = height;

    // get the imageData and pixel array from the canvas
    var imgData=ctx.getImageData(0, 0, width, height);
    var data=imgData.data;
    
    for (let i = 0; i < width * height; i++) {
        data[i * 4] = rgb.data[offset + i * 3]
        data[i * 4 + 1] = rgb.data[offset + i * 3 + 1]
        data[i * 4 + 2] = rgb.data[offset + i * 3 + 2]
        data[i * 4 + 3] = 255
    }

    if (!!click) {
        const offset = 40 * 4 * click.offsetY + click.offsetX * 4
        data[
            offset
        ] = 255;
        data[offset + 1] = 0
        data[offset + 2] = 255
        data[offset + 3] = 255
    }

    ctx.putImageData(imgData, 0, 0)
}

async function run (steps, initialState) {
    if(!session) session = await ort.InferenceSession.create('./salamander.onnx')
    
    const results = await session.run({
        steps: new ort.Tensor('int64', BigInt64Array.from([BigInt(steps)]), [1]),
        batch_size: new ort.Tensor('int64', BigInt64Array.from([BigInt(1)]), [1]),
        initial_state: new ort.Tensor('float32', initialState , [1, 40, 40, 16])
    })

    return results
}

function drawNext (rgb, step, next) {
    const nextStep = step < rgb.dims[0] - 1 ? step + 1 : 0;
    drawCanvas(rgb, step)
    if (nextStep != 0) {
        setTimeout(() => drawNext(rgb, nextStep, next), 100)
    } else {
        next()
    }
}

// Takes a state: OrtTensor, center point {x: int, y: int}, and a size x: int
function damage(state, center, size) {
    const halfSize = Math.round(size / 2)
    const xRange = [center.x - halfSize, center.x + halfSize]
    const yRange = [center.y - halfSize, center.y + halfSize]
    const [yDim, xDim, stride] = state.dims.slice(-3)
    
    let offset = 0;
    for (let y = yRange[0]; y <= yRange[1] && y < yDim; y++) {
        for (let x = xRange[0]; x <= xRange[1] && x < xDim; x++) {
            offset = (y * xDim + x) * stride
            for (let k = 0; k < stride; k++) {
                state.data[offset + k] = 0;
            }
        }
    }
}

function loop(state) {
    state = state || new Float32Array(1 * 40 * 40 * 16);
    const iters = 1
    run(iters, state).then((result) => {
        //drawNext(result.rgb, 0)
        const nextState = result.last_state
        if (!!click) {
            damage(nextState, {y: click.offsetY, x: click.offsetX}, 10)
            click = null
        }
        drawNext(result.rgb, 0, () => loop(nextState.data))
        //setTimeout(() => loop(nextState), 200)
        divIters.innerText = parseInt(divIters.innerHTML, 10) + iters
    })
}

function init () {
    document.getElementById("canvas").addEventListener('click', (e) => {
        click = e;
    })
    loop()
}

init()