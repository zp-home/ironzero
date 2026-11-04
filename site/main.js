// SPDX-License-Identifier: Apache-2.0
(() => {
  const navToggle = document.querySelector(".nav-toggle");
  const siteNav = document.querySelector(".site-nav");
  const year = document.querySelector("#year");
  const heroVisual = document.querySelector(".hero-visual");

  if (year) {
    year.textContent = String(new Date().getFullYear());
  }

  if (navToggle && siteNav) {
    navToggle.addEventListener("click", () => {
      const isOpen = siteNav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(isOpen));
    });

    siteNav.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        siteNav.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  document.querySelectorAll(".reveal").forEach((item, index) => {
    item.style.animationDelay = `${Math.min(index % 5, 4) * 55}ms`;
  });

  if (heroVisual && window.matchMedia("(pointer: fine)").matches) {
    heroVisual.addEventListener("pointermove", (event) => {
      const rect = heroVisual.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width - 0.5;
      const y = (event.clientY - rect.top) / rect.height - 0.5;
      heroVisual.style.setProperty("--tilt-x", `${y * -2.2}deg`);
      heroVisual.style.setProperty("--tilt-y", `${x * 2.2}deg`);
    });

    heroVisual.addEventListener("pointerleave", () => {
      heroVisual.style.setProperty("--tilt-x", "0deg");
      heroVisual.style.setProperty("--tilt-y", "0deg");
    });
  }
})();
