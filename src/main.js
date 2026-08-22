import { Game } from './game.js';

const renderDiv = document.getElementById('renderDiv');

if (!renderDiv) {
  console.error('Fatal Error: renderDiv element not found.');
} else {
  const game = new Game(renderDiv);
  game.start();
}
