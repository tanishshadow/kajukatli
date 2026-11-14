(() => {
  const nav = document.querySelector('.site-nav');
  const button = document.querySelector('.menu-toggle');
  const links = document.querySelector('.nav-links');
  if (nav) window.addEventListener('scroll', () => nav.classList.toggle('is-scrolled', window.scrollY > 24), { passive: true });
  if (button && links) {
    button.addEventListener('click', () => {
      const open = button.getAttribute('aria-expanded') !== 'true';
      button.setAttribute('aria-expanded', String(open));
      button.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
      links.classList.toggle('is-open', open);
    });
    links.addEventListener('click', event => {
      if (event.target.closest('a')) {
        links.classList.remove('is-open');
        button.setAttribute('aria-expanded', 'false');
        button.setAttribute('aria-label', 'Open navigation');
      }
    });
  }
  const targets = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.add('is-visible'); observer.unobserve(entry.target); }
    }), { threshold: .15 });
    targets.forEach(target => observer.observe(target));
  } else targets.forEach(target => target.classList.add('is-visible'));
})();
