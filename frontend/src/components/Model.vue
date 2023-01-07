<script setup lang="ts">
import { ref, onMounted, Ref, reactive } from 'vue'

import { run, drawNext, damage } from '../helpers/main'

// Define props
const props = defineProps<{ modelFile: string }>()



// Reactive state
const state = reactive({
  iterations: 0
})

// Refs
let canvas: Ref<HTMLCanvasElement | null>  = ref(null);
let context: CanvasRenderingContext2D | null | undefined = undefined;


let session = await ort.InferenceSession.create(`./models/${props.modelFile}`)


let click: MouseEvent | null = null
let reset = false

function loop(cellState: Float32Array | null, session: ort.InferenceSession, canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) {
    if (reset) {
      reset = false
      cellState = null
    }
    cellState = cellState || new Float32Array(1 * 40 * 40 * 16);
    const iters = 1
    run(iters, cellState, session).then((result) => {
        //drawNext(result.rgb, 0)
        const nextState = result.last_state
        if (!!click) {
            damage(nextState, {y: Math.round(40 * click.offsetY / (<any>click.target).clientHeight), x: Math.round(40 * click.offsetX / (<any>click.target).clientWidth)}, 10)
            click = null
        }
        drawNext(
            result.rgb,
            0,
            () => loop(nextState.data, session, canvas, context),
            canvas,
            context
        )
        state.iterations += iters
    })
}

// Event handlers
function canvasClick (e: MouseEvent) {
  click = e
}

function doReset () {
  reset = true
}

// Lifecycle hooks
onMounted(() => {
  context = canvas.value?.getContext('2d')
  if(canvas.value && context)
    loop(null, session, canvas.value, context)
})
</script>

<template>
  <div class="container">
      <div>
        <canvas class="canvas" width="40" height="40" ref="canvas" @click="canvasClick">
        </canvas>
      </div>
      <div>
        <div class="iters">{{state.iterations}}</div>
        <button @click="doReset()">reset</button>
      </div>
  </div>
</template>

<style scoped>
  .canvas {
      width: 160px;
      height: 160px;
  }
</style>
