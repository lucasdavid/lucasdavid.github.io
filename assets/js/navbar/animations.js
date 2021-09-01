var
  LG_WIDTH = 992,
  INVERT_AFTER_Y = 150;

var navbarAnimator = {
  mainNav: document.getElementById('mainNav'),
  navBrand: document.getElementById('mainNavBrand'),
  bar: document.getElementById('readProgressBar'),
  sideBar: document.getElementById('sidebar'),
  navBarInverted: false
};

function navbarInvertAnimated() {
  var winScroll = document.body.scrollTop || document.documentElement.scrollTop;

  if (navbarAnimator.bar) {
    var height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    var scrolled = (winScroll / height) * 100;

    navbarAnimator.bar.style.width = scrolled + "%";
  }

  var shouldInvert = winScroll > INVERT_AFTER_Y;

  if (shouldInvert == navbarAnimator.navBarInverted)
    return;

  if (shouldInvert) {
    if (navbarAnimator.navBrand) navbarAnimator.navBrand.hidden = true;
    if (navbarAnimator.mainNav) {
      navbarAnimator.mainNav.classList.remove('navbar-dark', 'bg-dark');
      navbarAnimator.mainNav.classList.add('navbar-light', 'bg-white', 'shadow-sm');
    }
  } else {
    if (navbarAnimator.navBrand) navbarAnimator.navBrand.hidden = false;
    if (navbarAnimator.mainNav) {
      navbarAnimator.mainNav.classList.remove('navbar-light', 'bg-white', 'shadow-sm');
      navbarAnimator.mainNav.classList.add('navbar-dark', 'bg-dark');
    }
  }

  navbarAnimator.navBarInverted = shouldInvert;
}

if (document.documentElement.clientWidth >= LG_WIDTH) {
  window.addEventListener('scroll', navbarInvertAnimated);
}
