"use strict";

for (const tabs of document.querySelectorAll(".paper-tabs")) {
  const buttons = [...tabs.querySelectorAll('[role="tab"]')];
  function select(button, focus=false) {
    for (const item of buttons) {
      const active = item === button;
      item.setAttribute("aria-selected",String(active));
      item.tabIndex = active ? 0 : -1;
      byId(item.getAttribute("aria-controls")).hidden = !active;
    }
    if (focus) button.focus();
  }
  buttons.forEach((button,index) => {
    button.addEventListener("click",() => select(button));
    button.addEventListener("keydown",event => {
      const next = event.key === "ArrowRight" ? (index+1)%buttons.length
        : event.key === "ArrowLeft" ? (index+buttons.length-1)%buttons.length
        : event.key === "Home" ? 0 : event.key === "End" ? buttons.length-1 : null;
      if (next !== null) {event.preventDefault();select(buttons[next],true);}
    });
  });
}
