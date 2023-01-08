

function drawCanvas (rgb: ort.TypedTensor<'uint8'>, index: number, canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) {
    
    // size the canvas to your desired image
    const images = rgb.dims[0]
    const width = rgb.dims[2]
    const height = rgb.dims[3]
    const channels = rgb.dims[4]
    const targetChannels = 4
    const pixelCount = width * height * channels
    const offset = index * width * height * channels
    /*canvas.width = width;
    canvas.height = height;*/

    // get the imageData and pixel array from the canvas
    var imgData = ctx.getImageData(0, 0, width, height);
    
    var data = imgData.data;
    
    for (let i = 0; i < width * height; i++) {
        data[i * 4] = rgb.data[offset + i * 3]
        data[i * 4 + 1] = rgb.data[offset + i * 3 + 1]
        data[i * 4 + 2] = rgb.data[offset + i * 3 + 2]
        data[i * 4 + 3] = 255
    }
    ctx.putImageData(imgData, 0, 0)
}

async function run (steps: number, initialState: Float32Array, session: ort.InferenceSession) {
    const results = await session.run({
        steps: new ort.Tensor('int64', BigInt64Array.from([BigInt(steps)]), [1]),
        batch_size: new ort.Tensor('int64', BigInt64Array.from([BigInt(1)]), [1]),
        initial_state: new ort.Tensor('float32', initialState , [1, 40, 40, 16])
    }) as {
        'rgb': ort.TypedTensor<'uint8'>,
        'last_state': ort.TypedTensor<'float32'>
    }

    return results
}

function drawNext (rgb: ort.TypedTensor<'uint8'>, step: number, next: CallableFunction, canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) {
    const nextStep = step < rgb.dims[0] - 1 ? step + 1 : 0;
    drawCanvas(rgb, step, canvas, context)
    if (nextStep != 0) {
        setTimeout(() => drawNext(rgb, nextStep, next, canvas, context), 50)
    } else {
        next()
    }
}

function damage(state: ort.Tensor, center: {x: number, y: number}, size: number) {
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



export { run, drawNext, damage }